from flask import Flask, render_template, request, send_from_directory
from rank_images import *
from summarization import *
import json


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
default_name = '0'
text = default_name

@app.route("/")
@app.route("/home")

def index():
    import os
    papers = os.listdir('static/papers')
    return render_template('home.html', title='About', data=papers)



@app.route("/formPDF",methods=['GET','POST'])
def formPDF():
    text = request.form.get('text',default_name)
    if text.__eq__(default_name):
        return render_template('formPDF.html')
    else:
        try:

            summary = summarize(text + '.tei.xml')

            section_text = ""
            for section,section_text1 in summary.items():
                section_text += '\n'+section.upper() + ":"+section_text1+'\n'

            with open(os.path.join('paper ' + text, 'ranking of images.txt')) as f:
                images_ranking = list()
                for line in f:
                    img_name = re.sub('\n', '', line.split(" ")[1]).strip()
                    if len(line) > 1:
                        images_ranking.append(img_name)

            json_files = os.listdir('json')

            file = 'json'+text+'.json'

            caption = ''

            image_name = images_ranking[0]
            with open('json/' + file) as json_file:
                text = json_file.read()
                json_data = json.loads(text)


                for i in json_data:
                    if image_name.__eq__(i['renderURL']):

                        caption = i['caption']
                    else:
                        caption = " "


                    break

            print('cap',caption)
            return render_template('summary.html', title='About', data=section_text,images = images_ranking[0],fig_caption=caption)


        except:
            print(text)
            return render_template('formPDF.html')




if __name__ == '__main__':
    app.run(port=4555, debug=True)