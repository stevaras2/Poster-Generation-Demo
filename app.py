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
    reports = os.listdir('static/reports')

    return render_template('home.html', title='Papers', data=reports)



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

            paper_title = get_paper_name(text + ".tei.xml")
            file = 'json'+text+'.json'

            caption = ''

            image_name = images_ranking[:3]

            image_section = pd.read_csv(os.path.join('paper ' + text, 'image_section.csv'), header=None)
            img_and_caption = dict()
            index = 0
            with open('json/' + file) as json_file:
                text = json_file.read()
                json_data = json.loads(text)
                for i in json_data:
                    if i['renderURL'] in image_name:
                        img_and_caption[i['renderURL']] = [i['caption'],image_section.iloc[index,1]]

                       # print(image_section.iloc[index,1])
                        index += 1

            for k,v in img_and_caption.items():
                print(k,v)
            return render_template('summary.html', title='About', data=section_text,paper_title=paper_title,
                                   images = img_and_caption)
        except:
            print(text)
            return render_template('formPDF.html')




if __name__ == '__main__':
    app.run(port=4555, debug=True)