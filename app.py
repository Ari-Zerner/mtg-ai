from flask import Flask, request, render_template, jsonify
import mtgai
import markdown
import time
import uuid
import threading

# Global dictionary to track job progress and results
jobs = {}

# Clean up completed jobs older than 1 hour
def cleanup_old_jobs():
    current_time = time.time()
    to_delete = []
    for job_id, job in jobs.items():
        if job.get('completed') and current_time - job.get('timestamp', 0) > 3600:
            to_delete.append(job_id)
    for job_id in to_delete:
        del jobs[job_id]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    formats = mtgai.get_format_list()
    return render_template('index.html', formats=formats)

@app.route('/', methods=['POST'])
def submit_deck():
    decklist = request.form['decklist']
    format_value = request.form['format']
    additional_info = request.form['additional_info']
    
    # Create a unique job id and record initial progress
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"progress": ["Report ID: " + job_id], "completed": False, "result": None, "timestamp": time.time()}
    
    # Background thread to run deck analysis
    def run_job(job_id, decklist, format_value, additional_info):
        # Define progress callback to update job progress
        def progress_update(message, is_update=False):
            if is_update:
                jobs[job_id]["progress"].pop()
            jobs[job_id]["progress"].append(message)
        try:
            # The get_deck_advice function now takes a progress_callback
            advice = mtgai.get_deck_advice(decklist, format=format_value, additional_info=additional_info, progress_callback=progress_update)
            decklist_section = f"## Decklist\n{decklist}"
            additional_info_section = f"## Additional Info\n{additional_info}" if additional_info.strip() else ""
            format_section = f"## Format\n{format_value}" if format_value.strip() else ""
            report = f"# Advice\n{advice}\n\n# Provided Information\n{decklist_section}\n{format_section}\n{additional_info_section}".replace('\n', '  \n')
            result = {"content_md": report, "content_html": markdown.markdown(report)}
            jobs[job_id]["result"] = result
            jobs[job_id]["completed"] = True
            jobs[job_id]["progress"].append("Deck analysis complete. Report generated.")
        except Exception as e:
            jobs[job_id]["progress"].append("Error encountered: " + str(e))
            jobs[job_id]["completed"] = True
    
    thread = threading.Thread(target=run_job, args=(job_id, decklist, format_value, additional_info))
    thread.start()
    
    # Render a progress page that polls the /status endpoint
    return render_template('progress.html', job_id=job_id)

@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    if job_id in jobs:
        return jsonify({"progress": jobs[job_id]["progress"], "completed": jobs[job_id]["completed"]})
    else:
        return jsonify({"progress": [f"Job {job_id} not found"], "completed": True}), 404

@app.route('/report/<job_id>', methods=['GET'])
def get_report(job_id):
    if job_id not in jobs:
        return "Job not found.", 404
    
    if not jobs[job_id]["completed"]:
        return "Job still in progress. Please refresh later.", 202
    
    result = jobs[job_id]["result"]
    if result is None:
        # Handle the case where the job completed with an error
        error_messages = [msg for msg in jobs[job_id]["progress"] if "Error encountered:" in msg]
        error_message = error_messages[-1] if error_messages else "An unknown error occurred"
        return render_template('error.html', error_message=error_message)
    
    return render_template('report.html', content_html=result["content_html"], content_md=result["content_md"])

if __name__ == '__main__':
    app.run(debug=True)
