from flask import Flask, request, render_template
import mtgai
import markdown
import asyncio
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def report():
    decklist = request.form['decklist']
    format = request.form['format']
    additional_info = request.form['additional_info']
    
    advice = asyncio.run(mtgai.get_deck_advice(decklist, format=format, additional_info=additional_info))

    decklist_section = f'''
## Decklist
{decklist}
    '''
    
    additional_info_section = f'''
## Additional Info
{additional_info}
    ''' if additional_info else ''
    
    format_section = f'''
## Format
{format}
    ''' if format else ''
    
    report = f'''# Advice
{advice}

# Provided Information
{decklist_section}
{format_section}
{additional_info_section}
    '''.replace('\n', '  \n')
    
    return render_template('report.html', content_html=markdown.markdown(report), content_md=report)

if __name__ == '__main__':
    app.run(debug=True)
