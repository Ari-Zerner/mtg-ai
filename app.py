from flask import Flask, request, render_template
import mtgai
import markdown

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    advice = None
    if request.method == 'POST':
        decklist = request.form['decklist']
        format = request.form['format']
        additional_info = request.form['additional_info']
        
        # Get the advice
        advice_text = mtgai.get_deck_advice(decklist, format=format, additional_info=additional_info)
        
        # Convert to HTML
        advice = markdown.markdown(advice_text)
    
    return render_template('index.html', advice=advice)

if __name__ == '__main__':
    app.run(debug=True)
