import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.externals import joblib
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input

def rank_images_of_paper(paper_folder,figure_caption_dataset,bert_of_images,stacking_model,list_of_models):
    '''
    It ranks the images of a paper based on the probability that the stacking model predicts. It saves this ranking in
    the image_ranking.txt
    :param paper_folder: the folder of the paper that the  user would like to rank its images
    :param figure_caption_dataset: the path of the file 'figure_caption_dataset.csv'
    :param bert_of_images: the path of the file 'bert_figure_caption.csv'
    :param stacking_model: the pickle file that are store the stacking model
    :param list_of_models: a list with the pickle files of the base classifier that are used in the stacking model
    the correct order of the pickle file is(they are stored in the image_classification models folder):
    1)'InceptionV3_logistic.pkl',
    2)'InceptionV3_BERT_logistic.pkl'
    3)'InceptionV3_SVM.pkl'
    4)'InceptionV3_BERT_SVM.pkl'
    5)'InceptionResNetV2_logistic.pkl'
    6)'InceptionResNetV2_BERT_logistic.pkl'
    7)'InceptionResNetV2_SVM.pkl'
    8)'InceptionResNetV2_BERT_SVM.pkl'
    :return: None
    '''
    figure_caption = pd.read_csv(figure_caption_dataset, encoding='latin-1', header=None)
    bert_of_caption = pd.read_csv(bert_of_images, encoding='latin-1', header=None)
    bert_of_caption_dict = dict()# save the BERT embedding of each caption

    for row in bert_of_caption.iterrows():
        bert = row[1][1].replace('[', '')
        bert = bert.replace(']', '')
        bert_vector = np.fromstring(bert, dtype=np.float, sep=' ')
        bert_vector = list(bert_vector)
        #bert_vector_list.append(bert_vector)
        bert_of_caption_dict[row[1][0]] = np.asarray(bert_vector)

    #get the captions of the figures of the paper
    paper_figure_caption = dict()#dict(stores information only for the paper) with key the figure name and value its
    # caption's BERT embedding

    for row in figure_caption.iterrows():
        if row[1][0].split("-")[0][4:].__eq__(paper_folder.split(" ")[1]):
            if row[1][1] not in bert_of_caption_dict:
                paper_figure_caption[row[1][0]] = np.zeros(768)
            else:
                paper_figure_caption[row[1][0]] = bert_of_caption_dict[row[1][1]]

    images_list = os.listdir(paper_folder + '/paper_images')
    inception_resnetv2_model = InceptionResNetV2(weights='imagenet', include_top=False)


    im_des_dict_incResV2 = dict()
    i = 0
    size = len(images_list)
    for im in images_list:
        img_path = paper_folder+'/paper_images' + '/' + im
        img = image.load_img(img_path, target_size=(229, 229))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        inception_resnetv2_feature = inception_resnetv2_model.predict(img_data)
        im_des_dict_incResV2[im] = inception_resnetv2_feature
        i += 1
        print('Extracted features for ', i, "/", size, ' figures')
        print(inception_resnetv2_feature.shape)

    im_des_dict_incResV2_1 = dict()  # averaged features vectors
    inception_resnetv2_features = list()# the feature that will be fed into the classifiers that do not use BERT
    inception_resnetv2_features_BERT = list()# the feature that will be fed into the classifiers that do not use BERT
    # average to the feature vector in order to feed them in a Logistic Regression or/and SVM
    for im, feature_vector in im_des_dict_incResV2.items():
        im_des_dict_incResV2_1[im] = np.average(feature_vector, axis=(0, 1, 2))
        inception_resnetv2_features.append(im_des_dict_incResV2_1[im])
        print(im_des_dict_incResV2_1[im].reshape((1,1536)).shape,paper_figure_caption[im].reshape((1,768)).shape)
        inception_resnetv2_features_BERT.append(np.column_stack([im_des_dict_incResV2_1[im].reshape((-1,1536)),
                                                                 paper_figure_caption[im].reshape((-1,768))]))


    inception_resnetv2_features = np.asarray(inception_resnetv2_features)
    print(inception_resnetv2_features.shape)
    inception_resnetv2_features_BERT = np.average(np.asarray(inception_resnetv2_features_BERT),axis=1)
    print(inception_resnetv2_features_BERT.shape)

    inceptionv3_model = InceptionV3(weights='imagenet', include_top=False)

    im_des_dict_incV3 = dict()
    i = 0
    size = len(images_list)
    for im in images_list:
        img_path = paper_folder + '/paper_images' + '/' + im
        img = image.load_img(img_path, target_size=(229, 229))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        inceptionV3_feature = inceptionv3_model.predict(img_data)
        im_des_dict_incV3[im] = inceptionV3_feature
        i += 1
        print('Extracted features for ', i, "/", size, ' figures')
        print(inceptionV3_feature.shape)

    im_des_dict_incV3_1 = dict()  # averaged features vectors
    inceptionv3_features = list()  # the feature that will be fed into the classifiers that do not use BERT
    inceptionv3_features_BERT = list()  # the feature that will be fed into the classifiers that do not use BERT
    # average to the feature vector in order to feed them in a Logistic Regression or/and SVM
    for im, feature_vector in im_des_dict_incV3.items():
        im_des_dict_incV3_1[im] = np.average(feature_vector, axis=(0, 1, 2))
        inceptionv3_features.append(im_des_dict_incV3_1[im])

        inceptionv3_features_BERT.append(np.column_stack([im_des_dict_incV3_1[im].reshape((-1, 2048)),
                                                                 paper_figure_caption[im].reshape((-1, 768))]))

    inceptionv3_features = np.asarray(inceptionv3_features)
    print(inceptionv3_features.shape)
    inceptionv3_features_BERT = np.average(np.asarray(inceptionv3_features_BERT), axis=1)
    print(inceptionv3_features_BERT.shape)


    #load the base models
    inceptionV3_logistic = joblib.load(list_of_models[0])
    inceptionV3_BERT_logistic = joblib.load(list_of_models[1])
    inceptionV3_svm = joblib.load(list_of_models[2])
    inceptionV3_BERT_svm = joblib.load(list_of_models[3])
    inception_resnetv2_logistic = joblib.load(list_of_models[4])
    inception_resnetv2_BERT_logistic = joblib.load(list_of_models[5])
    inception_resnetv2_svm = joblib.load(list_of_models[6])
    inception_resnetv2_BERT_svm = joblib.load(list_of_models[7])


    #first model
    incv3_log_predict = inceptionV3_logistic.predict_proba(inceptionv3_features)
    inceptionv3_log_proba = list()
    #get the probability of an image to be included to a poster
    for proba in incv3_log_predict:
        inceptionv3_log_proba.append(proba[1])


    # second model
    incv3_BERT_log_predict = inceptionV3_BERT_logistic.predict_proba(inceptionv3_features_BERT)
    inceptionv3_BERT_log_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incv3_BERT_log_predict:
        inceptionv3_BERT_log_proba.append(proba[1])


    # third model
    incv3_svm_predict = inceptionV3_svm.predict_proba(inceptionv3_features)
    inceptionv3_svm_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incv3_svm_predict:
        inceptionv3_svm_proba.append(proba[1])


    # fourth model
    incv3_BERT_svm_predict = inceptionV3_BERT_svm.predict_proba(inceptionv3_features_BERT)
    inceptionv3_BERT_svm_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incv3_BERT_svm_predict:
        inceptionv3_BERT_svm_proba.append(proba[1])



    # fifth model
    incresv2_log_predict = inception_resnetv2_logistic.predict_proba(inception_resnetv2_features)
    inception_resnetv2_log_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incresv2_log_predict:
        inception_resnetv2_log_proba.append(proba[1])

    # sixth model
    incresv2_BERT_log_predict = inception_resnetv2_BERT_logistic.predict_proba(inception_resnetv2_features_BERT)
    inception_resnetv2_BERT_log_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incresv2_BERT_log_predict:
        inception_resnetv2_BERT_log_proba.append(proba[1])

    # seventh model
    incresv2_svm_predict = inception_resnetv2_svm.predict_proba(inception_resnetv2_features)
    inception_resnetv2_svm_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incresv2_svm_predict:
        inception_resnetv2_svm_proba.append(proba[1])

    # eighth model
    incresv2_BERT_svm_predict = inception_resnetv2_BERT_svm.predict_proba(inception_resnetv2_features_BERT)
    inception_resnetv2_BERT_svm_proba = list()
    # get the probability of an image to be included to a poster
    for proba in incresv2_BERT_svm_predict:
        inception_resnetv2_BERT_svm_proba.append(proba[1])


    #load stacking model

    stacking_model = joblib.load(stacking_model)

    features_fed_into_stacking = np.column_stack([
        np.asarray(inceptionv3_log_proba),
        np.asarray(inceptionv3_BERT_log_proba),
        np.asarray(inceptionv3_svm_proba),
        np.asarray(inceptionv3_BERT_svm_proba),
        np.asarray(inception_resnetv2_log_proba),
        np.asarray(inception_resnetv2_BERT_log_proba),
        np.asarray(inception_resnetv2_svm_proba),
        np.asarray(inception_resnetv2_BERT_svm_proba),
        ])

    stacking_predict = stacking_model.predict_proba(features_fed_into_stacking)

    stacking_proba = list()
    image_ranking = dict()
    for index,proba in enumerate(stacking_predict):
        stacking_proba.append(proba[1])
        image_ranking[images_list[index]] = proba[1]

    rank_images = sorted(image_ranking.items(), key=lambda x: x[1], reverse=True)

    with open(paper_folder+'/ranking of images.txt','w') as f:
        for index,rank in enumerate(rank_images):
            index += 1
            f.write(str(index)+") "+rank[0]+"\n")

if __name__ == '__main__':
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

    paper_list = ['paper 243',
                  'paper 248',
                  'paper 249',
                  'paper 251',
                  'paper 254',
                  'paper 255',
                  'paper 256',
                  'paper 258',
                  'paper 261',
                  'paper 281',
                  'paper 296',
                  'paper 370',
                  'paper 400'
                  ]
    #for paper in paper_list:
    rank_images_of_paper('paper 400', 'figure_caption_dataset.csv', 'bert_figure_caption.csv',
                             'stacking_model.pkl',
                             list_of_models)
