from rank_images import *
from summarization import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


if __name__ == '__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paper", required=True, help="add paper to create its poster")

    args = vars(ap.parse_args())

    paper = args['paper']

    summary = summarize(paper+'.tei.xml')
    for k,v in summary.items():
        print(v)
    '''
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



    rank_images_of_paper('paper '+paper, 'figure_caption_dataset.csv', 'bert_figure_caption.csv',
                         'stacking_model.pkl',
                         list_of_models)


    with open(os.path.join('paper '+paper,'ranking of images.txt')) as f:

        for line in f:
            img_name = re.sub('\n','',line.split(" ")[1]).strip()
            if len(line) > 1:
                img = mpimg.imread(os.path.join('paper '+paper,'paper_images',img_name))
                imgplot = plt.imshow(img)
                plt.show()
'''