from flask import Flask, render_template, request, send_from_directory
from rank_images import *
from summarization import *


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

            models = ['InceptionV3_logistic.pkl',
                      'InceptionV3_BERT_logistic.pkl',
                      'InceptionV3_SVM.pkl',
                      'InceptionV3_BERT_SVM.pkl',
                      'InceptionResNetV2_logistic.pkl',
                      'InceptionResNetV2_BERT_logistic.pkl',
                      'InceptionResNetV2_SVM.pkl',
                      'InceptionResNetV2_BERT_SVM.pkl']

            list_of_models = list()
            for model in models:
                list_of_models.append('image_classification models/' + model)

            rank_images_of_paper('paper ' + text, 'figure_caption_dataset.csv', 'bert_figure_caption.csv',
                                 'stacking_model.pkl',
                                 list_of_models)

            with open(os.path.join('paper ' + text, 'ranking of images.txt')) as f:
                images_ranking = list()
                for line in f:
                    img_name = re.sub('\n', '', line.split(" ")[1]).strip()
                    if len(line) > 1:

                        images_ranking.append(img_name)
            return render_template('summary.html', title='About', data=summary,images = images_ranking)


        except :
            print(text)
            return render_template('formPDF.html')




if __name__ == '__main__':
    app.run(port=4555, debug=True)