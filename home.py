from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance_matrix
from nltk.corpus import stopwords
from googlesearch import search
from datetime import timedelta
from bs4 import BeautifulSoup
from fcmeans import FCM
import pandas as pd
import numpy as np
import threading
import requests
import zeyrek
import nltk
import time
import copy
import re


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# HOME PAGE OF WEBSITE
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


# RESULT PAGE OF WEBSITE
@app.route("/result", methods=["GET", "POST"])
def web_page_clustering():

    # TO MEASURE PROGRAM EXECUTION TIME
    start_time_total = time.monotonic()

    result_clusters_output = {}
    search_key = ""
    number_sites = 0

    if request.method == "POST":

        start_time = time.monotonic()

        # WEB SCRAPING
        search_key = request.form.get("search")
        print("\nSearch key: " + search_key)

        all_websites = []
        only_urls = []
        # pause parameter's ideal value is 2.0, greater or less values may come up with errors or problems
        # To increase similarity between search() and real user results, lang attribute is selected for Turkey
        for urls in search(search_key, lang='tr', num=25, start=0, stop=25):
            only_urls.append(urls)

        # Counters is a list that holds 3 value which initialized with 0. The first cell stores number of web pages,
        # the second one for unaccessed URLs, the third for number of non-found titles and the last one for non-found
        # description of web pages
        counters = [0, 0, 0, 0]

        # FUNCTION TO SCRAP ALL INFORMATION FROM A URL USING THE SAME REQUEST SESSION
        def scrap_urls(url, requests_session):

            try:

                response = requests_session.get(url)
                soup = BeautifulSoup(response.content, 'lxml', from_encoding=response.encoding)


                title_tag = soup.find('title')
                if title_tag is not None:
                    title_temp = title_tag.get_text()
                else:
                    title_selectors = [
                        {"name": "title"},
                        {"name": "og:title"},
                        {"property": "title"},
                        {"property": "og:title"}
                    ]

                    for selector in title_selectors:
                        title_tag = soup.find(attrs=selector)
                        if title_tag and title_tag.get('content'):
                            title_temp = title_tag['content']
                            break
                    else:
                        counters[2] += 1
                        title_temp = ""


                description_selectors = [
                    {"name": "description"},
                    {"name": "og:description"},
                    {"property": "description"},
                    {"property": "og:description"}
                ]

                for selector in description_selectors:
                    description_tag = soup.find(attrs=selector)
                    if description_tag and description_tag.get('content'):
                        desc_temp = description_tag['content']
                        break
                else:
                    counters[3] += 1
                    desc_temp = ""

                # Only when both title and description are scraped, that website is added to process
                if title_temp != "" and desc_temp != "":
                    all_websites.append({'url': url, 'title': title_temp, 'description': desc_temp})
                    counters[0] += 1


            except Exception:
                counters[1] += 1

            return

        # FUNCTION USES MULTI-THREADING TO START SCRAPING ALL PAGES AT THE SAME TIME
        def parallel_scrap(request_session):

            threads = [threading.Thread(target=scrap_urls, args=(url, request_session))
                       for url in only_urls]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            print("\nNumber of scraped websites                          :", counters[0])
            print("Number of websites that couldn't be accessed        :", counters[1])
            print("Number of websites whose title not found            :", counters[2])
            print("Number of websites whose description not found      :", counters[3])

            return

        # By using Session(), one connection is set and it is re-used
        session = requests.Session()
        # get_data(search_key, session)
        parallel_scrap(session)
        # Storing web pages number in a variable to use later
        number_sites = counters[0]
        # This line is added to show the row and preprocessed data together, to compare
        all_websites_row = copy.deepcopy(all_websites)

        # TIME MEASUREMENT OF WEB SCRAPING
        end_time = time.monotonic()
        time_scraping = timedelta(seconds=end_time - start_time)
        start_time = time.monotonic()

        # DATA PREPROCESSING
        # item represents index in related data structure, like index, to follow the code better
        def remove_duplicate():

            for item in all_websites:
                # Removed duplicate words in each website's data with tokenization
                temp = []
                title_tokens_temp = nltk.word_tokenize(item['title'])
                [temp.append(word) for word in title_tokens_temp if word not in temp]
                item['title'] = TreebankWordDetokenizer().detokenize(temp)

                temp = []
                desc_tokens_temp = nltk.word_tokenize(item['description'])
                [temp.append(word) for word in desc_tokens_temp if word not in temp]
                item['description'] = TreebankWordDetokenizer().detokenize(temp)

            return

        def remove_duplicate_noise():

            for item in all_websites:
                # Removed duplicate words in each website's data with tokenization
                temp = []
                title_tokens_temp = nltk.word_tokenize(item['title'])
                # len(word) < 16 condition allows to avoid weird words after removal of punctuations, such as url adress
                # Lemmatization is used to find the root of words, thus max 15 character length is chosen for tokens
                [temp.append(word) for word in title_tokens_temp if word not in temp and 1 < len(word) < 16]
                item['title'] = TreebankWordDetokenizer().detokenize(temp)

                temp = []
                desc_tokens_temp = nltk.word_tokenize(item['description'])
                [temp.append(word) for word in desc_tokens_temp if word not in temp and 1 < len(word) < 16]
                item['description'] = TreebankWordDetokenizer().detokenize(temp)

            return

        def preprocessing():

            for item in all_websites:
                # Turkish special letters are added to Regex expression to keep exact Turkish words
                # Example: AĞIZ -> ağız, not agiz
                turkish_chars_space = " çğıöşü"

                # Lowered case, removed whitespace, punctuations and numbers
                item['title'] = item['title'].lower()
                item['title'] = re.sub(r'[^a-z' + turkish_chars_space + ']', '', item['title'])
                item['title'] = item['title'].strip()

                item['description'] = item['description'].lower()
                item['description'] = re.sub(r'[^a-z' + turkish_chars_space + ']', '', item['description'])
                item['description'] = item['description'].strip()

            return

        preprocessing()
        remove_duplicate()

        def lemmatization():

            analyzer = zeyrek.MorphAnalyzer()
            stop_words = set(stopwords.words('turkish'))
            counter_lemma = 0
            for item in all_websites:

                # Tokenization and stop words removal for titles
                title_tokens = nltk.word_tokenize(item['title'])
                stopped_words_title = [j for j in title_tokens if j not in stop_words]

                # Lemmatization for titles
                title_result = ''
                title_lemma = ''
                for words in stopped_words_title:
                    title_lemmas = analyzer.lemmatize(words)
                    try:
                        # The first lemma is used, the last index represents index of lemmas [0][1][0] <-
                        title_lemma = title_lemmas[0][1][0]
                    except Exception:
                        counter_lemma += 1

                    title_result += title_lemma + ' '
                # To ensure that lemma words are also in lower case
                title_result = title_result.lower()
                item['title'] = title_result

                # Tokenization and stop words removal for descriptions
                desc_tokens = nltk.word_tokenize(item['description'])
                stopped_words_desc = [j for j in desc_tokens if j not in stop_words]

                # Lemmatization for descriptions
                desc_result = ''
                desc_lemma = ''
                for words in stopped_words_desc:
                    desc_lemmas = analyzer.lemmatize(words)
                    try:
                        desc_lemma = desc_lemmas[0][1][0]
                    except Exception:
                        counter_lemma += 1

                    desc_result += desc_lemma + ' '
                desc_result = desc_result.lower()
                item['description'] = desc_result

            print("Number of words for which a Turkish lemma not found :", counter_lemma)

            return

        lemmatization()
        remove_duplicate_noise()

        # PRINTING ROW AND PREPROCESSED DATA
        def print_data():

            print("\n\nWep pages before and after row data preprocessing:\n")
            index_sites = 0
            for (item_row, item_new) in zip(all_websites_row, all_websites):
                print(index_sites, '.', item_row['url'])
                print("Title Row                :", item_row['title'])
                print("Title Preprocessed       :", item_new['title'])
                print("Description Row          :", item_row['description'])
                print("Description Preprocessed :", item_new['description'], "\n")
                index_sites += 1

            return

        print_data()

        # TIME MEASUREMENT OF DATA PREPROCESSING
        end_time = time.monotonic()
        time_preprocessing = timedelta(seconds=end_time - start_time)
        start_time = time.monotonic()

        # _title -> name style of titles' data variables & structures, ex: something_title
        # _desc -> name style of descriptions' data variables & structures, ex: something_desc
        # _title_desc -> name style of titles' and descriptions' joint data variables & structures, ex: smthn_title_desc
        # For example, bow_title_desc represents title-description joint bag of words
        # CREATION OF BAG OF WORDS, WHICH IS USED TO FIND WORDS' FREQUENCIES
        bow_title = {}
        bow_desc = {}
        bow_title_desc = {}

        def create_bow():

            for item in all_websites:

                # Bag of Words for titles
                # bow_title_desc's word frequency calculation is done in both title and description for loops
                title_tokens = nltk.word_tokenize(item['title'])
                for token in title_tokens:
                    if token not in bow_title.keys():
                        bow_title[token] = 1
                        bow_title_desc[token] = 1
                    else:
                        bow_title[token] += 1
                        bow_title_desc[token] += 1

                # Bag of Words for descriptions
                desc_tokens = nltk.word_tokenize(item['description'])
                for token in desc_tokens:
                    if token not in bow_desc.keys():
                        bow_desc[token] = 1
                        bow_title_desc[token] = 1
                    else:
                        bow_desc[token] += 1
                        bow_title_desc[token] += 1

            return

        create_bow()

        # CREATION OF VECTORS
        vectors_title = []
        vectors_desc = []
        vectors_title_desc = []

        def create_vectors():

            for item in all_websites:

                # Vectors of titles
                vector_title = []
                title_tokens = nltk.word_tokenize(item['title'])
                for token in bow_title:
                    if token in title_tokens:
                        vector_title.append(1)
                    else:
                        vector_title.append(0)
                vectors_title.append(vector_title)

                # Vectors of descriptions
                vector_desc = []
                desc_tokens = nltk.word_tokenize(item['description'])
                for token in bow_desc:
                    if token in desc_tokens:
                        vector_desc.append(1)
                    else:
                        vector_desc.append(0)
                vectors_desc.append(vector_desc)

                # Vectors of title-description documents
                vector_title_desc = []
                title_tokens = nltk.word_tokenize(item['title'])
                desc_tokens = nltk.word_tokenize(item['description'])
                title_tokens.extend(desc_tokens)
                title_desc_tokens = title_tokens
                for token in bow_title_desc:
                    if token in title_desc_tokens:
                        vector_title_desc.append(1)
                    else:
                        vector_title_desc.append(0)
                vectors_title_desc.append(vector_title_desc)

            print("\nTitle vector size                          : ", len(bow_title))
            print("Description vector size                    : ", len(bow_desc))
            print("Title and description document vector size : ", len(bow_title_desc))

            return

        create_vectors()

        # CLUSTERING
        # FUNCTION TO PRINT RESULTS OF CLUSTERING, WITH CLUSTER NO AND URLs
        def print_clustering(labels, result_clusters):

            n = 0
            for item in labels:
                if item in result_clusters:
                    result_clusters[item].append(all_websites[n]['url'])
                else:
                    result_clusters[item] = [all_websites[n]['url']]
                n += 1

            for item in result_clusters:
                print("Cluster ", item, ':')
                for url in result_clusters[item]:
                    print(url)

            return

        # FUNCTION TO FIND SILHOUETTE SCORES ACCORDING TO 'K' CLUSTER NUMBER
        def find_ks(data, results):

            # Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1 (document.)
            ks_range = range(2, number_sites - 1)
            for kth in ks_range:
                k_model = KMeans(n_clusters=kth)
                k_model.fit(data)
                preds = k_model.predict(data)
                score = silhouette_score(data, preds)

                # result[0] is cluster number, [1] is max score (float)
                if results[1] < score:
                    results[1] = score
                    results[0] = kth

            print("Cluster number         :", results[0], "\nMax Silhouette score   : {:.2f}".format(results[1]))

            return

        # FUNCTION TO CLUSTER WITH K-MEANS AND BOW
        def kmeans_bow(vectors, len_bow, result_clusters_bow):

            # Data is unclustered row data from the internet, websites and their title - descriptions. To make it a
            # proper data set, coloumns added as Word 1    Word 2    Word 3.. To analyse data set better
            df = pd.DataFrame(vectors, columns=[f'Word{x + 1}' for x in range(len_bow)])
            df_scaled = StandardScaler().fit_transform(df)

            result = [0, 0]
            find_ks(df_scaled, result)
            n_clusters = result[0]

            model = KMeans(n_clusters)
            labels = model.fit_predict(df_scaled)
            centroids = model.cluster_centers_

            # Masurement of each samples' distance to associated cluster
            sample_distances = pd.DataFrame(distance_matrix(df_scaled, centroids))
            mean_distance = sample_distances.mean().to_string()
            print("Samples' distance means to centroids :")
            print(mean_distance, "\n")

            print_clustering(labels, result_clusters_bow)

            return

        # FUNCTION TO CLUSTER WITH K-MEANS AND DOC2VEC
        def kmeans_doc2vec(which_data, result_clusters_doc2vec):

            documents = []

            if which_data == "Title":

                for item in all_websites:
                    document = item['title']
                    documents.append(document)

            elif which_data == "Description":

                for item in all_websites:
                    document = item['description']
                    documents.append(document)

            else:

                # Title-description documents created, joint data expression is replaced by documents now
                for item in all_websites:
                    temp = []
                    document = item['title'] + " " + item['description']
                    doctoken = nltk.word_tokenize(document)
                    [temp.append(word) for word in doctoken if word not in temp]
                    document = TreebankWordDetokenizer().detokenize(temp)
                    documents.append(document)

            tokenized = []
            for d in documents:
                tokenized.append(nltk.word_tokenize(d))
            tagged_data = [TaggedDocument(d, [j]) for j, d in enumerate(tokenized)]

            d2vmodel = Doc2Vec(tagged_data, vector_size=len(tagged_data), window=2, min_count=1, workers=4, epochs=100)
            d2vmodel.save("test_doc2vec.model")
            d2vmodel = Doc2Vec.load("test_doc2vec.model")

            doc2vectors = []
            for idx in range(number_sites):
                doc2vectors.append(d2vmodel.dv[idx])
            doc2vectors_scaled = StandardScaler().fit_transform(doc2vectors)

            result = [0, 0]
            find_ks(doc2vectors_scaled, result)
            n_clusters = result[0]

            # To compare K-Means with FCM and GM, the same number of 'k' is specified for the latter two algorithms
            if which_data == "Both":
                common_k_number[0] = n_clusters

            model = KMeans(n_clusters=n_clusters).fit(doc2vectors_scaled)
            labels = model.predict(doc2vectors_scaled)
            centroids = model.cluster_centers_

            sample_distances = pd.DataFrame(distance_matrix(doc2vectors_scaled, centroids))
            mean_distance = sample_distances.mean().to_string()
            print("Samples' distance means to centroids :")
            print(mean_distance, "\n")

            print_clustering(labels, result_clusters_doc2vec)

            return doc2vectors_scaled

        # FUNCTION TO CLUSTER WITH FUZZY C-MEANS AND DOC2VEC
        def fuzzy_cmeans(result_clusters_fcm, scaled_data):

            np_doc = np.array(scaled_data)
            n_clusters = common_k_number[0]

            fcm = FCM(n_clusters=n_clusters)
            fcm.fit(np_doc)
            labels_fcm = fcm.predict(np_doc)

            print_clustering(labels_fcm, result_clusters_fcm)

            return np_doc

        # FUNCTION TO CLUSTER WITH GAUSSIAN MIXTURE MODEL AND DOC2VEC
        def gaussian_model(result_clusters_gm, np_data):

            n_clusters = common_k_number[0]
            gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(np_data)
            labels_gm = gm.predict(np_data)

            print_clustering(labels_gm, result_clusters_gm)

            return

        # FUNCTION TO CLUSTER WITH DBSCAN AND DOC2VEC
        def dbscan(result_clusters_dbscn, np_data):

            dbscn = DBSCAN()
            labels_dbscn = dbscn.fit_predict(np_data)

            print_clustering(labels_dbscn, result_clusters_dbscn)

            return

        # FUNCTION TO CLUSTER WITH HIERARCHICAL AND DOC2VEC
        def hierarchical(result_clusters_hierar, np_data):

            n_clusters = common_k_number[0]
            hierar = AgglomerativeClustering(n_clusters=n_clusters)
            labels_hierar = hierar.fit_predict(np_data)

            print_clustering(labels_hierar, result_clusters_hierar)

            return

        # This variable stores the common k cluster number to use for Fuzzy CM and Gaussian M as well,
        # to compare them with K-Means algorithm
        common_k_number = [0]

        # K-MEANS CLUSTERING WITH BOW FOR TITLES
        print("\n\n\nTitles BOW & K-means clustering")
        len_bow_title = len(bow_title)
        result_clusters_bow_title = {}
        kmeans_bow(vectors_title, len_bow_title, result_clusters_bow_title)

        # K-MEANS CLUSTERING WITH DOC2VEC FOR TITLES
        print("\n\n\nTitles doc2vec & K-means clustering")
        data_name = "Title"
        result_clusters_doc2vec_title = {}
        kmeans_doc2vec(data_name, result_clusters_doc2vec_title)

        # K-MEANS CLUSTERING WITH BOW FOR DESCRIPTIONS
        print("\n\n\nDescriptions BOW & K-means clustering")
        len_bow_desc = len(bow_desc)
        result_clusters_bow_desc = {}
        kmeans_bow(vectors_desc, len_bow_desc, result_clusters_bow_desc)

        # K-MEANS CLUSTERING WITH DOC2VEC FOR DESCRIPTIONS
        print("\n\n\nDescriptions doc2vec & K-means clustering")
        data_name = "Description"
        result_clusters_doc2vec_desc = {}
        kmeans_doc2vec(data_name, result_clusters_doc2vec_desc)

        # K-MEANS CLUSTERING WITH BOW FOR TITLE-DESCRIPTION DOCUMENTS
        print("\n\n\nTitles and descriptions joint data BOW & K-means clustering")
        len_bow_title_desc = len(bow_title_desc)
        result_clusters_bow_title_desc = {}
        kmeans_bow(vectors_title_desc, len_bow_title_desc, result_clusters_bow_title_desc)

        # K-MEANS CLUSTERING WITH DOC2VEC FOR TITLE-DESCRIPTION DOCUMENTS
        print("\n\n\nTitles and descriptions documents data doc2vec & K-means clustering")
        data_name = "Both"
        result_clusters_doc2vec_title_desc = {}
        scaled_vectors_title_desc = kmeans_doc2vec(data_name, result_clusters_doc2vec_title_desc)

        # FUZZY C-MEANS CLUSTERING WITH DOC2VEC FOR TITLE-DESCRIPTION DOCUMENTS
        print("\n\n\nTitle and description documents doc2vec & Fuzzy C-means clustering")
        result_clusters_fuzzycm = {}
        np_vectors_title_desc = fuzzy_cmeans(result_clusters_fuzzycm, scaled_vectors_title_desc)

        # GAUSSIAN MIXTURE MODEL CLUSTERING WITH DOC2VEC FOR TITLE-DESCRIPTION DOCUMENTS
        print("\n\n\nTitle and description documents doc2vec & Gaussian mixture clustering")
        result_clusters_gaussian = {}
        gaussian_model(result_clusters_gaussian, np_vectors_title_desc)

        # DBSCAN CLUSTERING WITH DOC2VEC FOR TITLE-DESCRIPTION DOCUMENTS
        print("\n\n\nTitle and description documents doc2vec & DBSCAN clustering")
        result_clusters_dbscan = {}
        dbscan(result_clusters_dbscan, np_vectors_title_desc)

        # HIERARCHICAL CLUSTERING WITH DOC2VEC FOR TITLE-DESCRIPTION DOCUMENTS
        print("\n\n\nTitle and description documents doc2vec & hierarchical clustering")
        result_clusters_hierarchical = {}
        hierarchical(result_clusters_hierarchical, np_vectors_title_desc)

        # WEBSITE INTERFACE CLUSTERING METHOD'S SELECTION
        methods = {0: result_clusters_bow_title_desc, 1: result_clusters_doc2vec_title_desc, 2: result_clusters_fuzzycm,
                   3: result_clusters_gaussian}
        method_choice = int(request.form.get("methods"))
        result_clusters_output = methods.get(method_choice)

        # TIME MEASUREMENT OF CLUSTERING AND TOTAL OPERATION. PRINTING ALL MAIN PROCESSES' TIME INFORMATION
        end_time = time.monotonic()
        time_clustering = timedelta(seconds=end_time - start_time)
        end_time_total = time.monotonic()
        time_total = timedelta(seconds=end_time_total - start_time_total)
        print("\n\nWeb scraping time         :", time_scraping, "seconds")
        print("Data preprocessing time   :", time_preprocessing, "seconds")
        print("Clustering operation time :", time_clustering, "seconds")
        print("Execution time of program :", time_total, "seconds\n\n")


    return render_template("result.html", search_key=search_key, web_pages_number=number_sites,
                           result_clusters=result_clusters_output)


if __name__ == "__main__":
    app.run(debug=True)
