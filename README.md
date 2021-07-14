# Turkish Web-Page-Clustering
Web page clustering according to content with Python implementation.


### Contents of document
1. Introduction
2. Dataset
3. Operation Steps
4. Methods, Languages and Platforms
5. How to install and use the project
6. Screenshots of the program
7. Some Problems


### 1. Introduction
In this study a system with website interface provides search operation for users and it clusters the results from search engine. With Data Mining and Natural Language Processing operations the system enables users to search in a grouped way.


### 2. Dataset
The project uses web pages' title and description data from each website that are scraped. To compare clustering operation, 3 data sets are used. These are "only title", "only description" and "title-description joint" data. 

Note: Web pages number to be scraped is 25 but due to some scraping issues, the operation may not be completed with 25 web pages, it's below 25 most of the time.


### 3. Operation Steps
- Search by users
- Getting results from search engine
- Text preprocessing with Tokenization and Lemmatization
- Vector transformation with Bag of Words
- Vector transformation with Doc2Vec
- Clustering with K-Means, Fuzzy C-Means and Gaussian Mixture Model algorithms
- Showing the results via the website interface

The steps with input/output are shown in the diagram below.

![process](https://user-images.githubusercontent.com/57035819/124019508-d3e0c180-d9f1-11eb-8105-ebaac8ace4ca.png)


### 4. Methods, Languages and Platforms
- Lemmatization
- Bag of Words
- Document to vector (D2V)
- K-Means
- Fuzzy C-Means
- Gaussian Mixture Model
- Elbow Method
- Silhouette Score
- Python with Flask framework
- Standard HTML, CSS and JavaScript
- PyCharm IDE


### 5. How to install and use the project

A new project on PyCharm IDE should be created and the source code should be located in the new project's folder. After installing all necessary packages, by runing home.py file, the website will be opened and the system will be ready to be used. It requires two input; one is the method of clustering algorithm with vector transformation method and the key that will be searched. The output of the system is again shown on the website, thus there is no need to look for any output file. All operations are done via the website in local machine.

Note: To use BeautifulSoup parser "lxml", lxml package should be installed as well (interpreter doesn't notify (no error in PyCharm) developer to install it, thus it's informed here).


### 6. Screenshots of website interface

#### Initial state
![home](https://user-images.githubusercontent.com/57035819/124016874-bfe79080-d9ee-11eb-8e6c-5ac1a5d6e30b.png)


#### Entering inputs
![input_method](https://user-images.githubusercontent.com/57035819/124016924-cd9d1600-d9ee-11eb-9458-ea41bfdb54a7.png)
![input_key](https://user-images.githubusercontent.com/57035819/124018954-1b1a8280-d9f1-11eb-870d-5eb19cdf5093.png)


#### Results of operations
![output](https://user-images.githubusercontent.com/57035819/124018576-aba49300-d9f0-11eb-8d6b-eb8512c3d083.png)
![output2](https://user-images.githubusercontent.com/57035819/124018595-afd0b080-d9f0-11eb-8cb0-10b4eb21ac33.png)


### 7. Some Problems
- Web scraping issues. Some methods to block information gain, some components that may make scraping fail.
- Although with multi-threading, scraping operation works faster comparing with the first version of the program but still it takes 15-30 secs to terminate (except the first run when the program started). Also clustering algorithms' operation time changes. 
- Turkish content is difficult to lemmatize and sometimes the relevant function that is used in the project may not work properly
- System success measurement is easy in classification studies but that is not the case in clustering. Sşnce there is no class label information of the data of this project, it's challenging to decide how good the program performed, the clustered operation done. Still, to have an approximation about the output, a metric to show the average distance of cluster points to centroid, which is called Davies Bouldin Index, is used in this project.


### Project Owner and Author
- [honourrable](https://github.com/honourrable)
