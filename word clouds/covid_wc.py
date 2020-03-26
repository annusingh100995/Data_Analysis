import nltk
import urllib.request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# https://en.wikipedia.org/wiki/2019%E2%80%9320_coronavirus_pandemic
response =  urllib.request.urlopen('https://en.wikipedia.org/wiki/Coronavirus_disease_2019')
pandemic = urllib.request.urlopen('https://en.wikipedia.org/wiki/2019%E2%80%9320_coronavirus_pandemic')
UK_covid = urllib.request.urlopen('https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_the_United_Kingdom')
INDIA_covid = urllib.request.urlopen('https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_India')
def word_cloud(response):
    covid_html = response.read()
    soup_covid = BeautifulSoup(covid_html, "lxml")
    covid_text = soup_covid.get_text(strip = True)
    #print(covid_text)
    covid_tokens = [t for t in covid_text.split()]
    #print(covid_tokens)
    sr= stopwords.words('english')
    clean_covid_tokens = covid_tokens[:]
    for token in covid_tokens:
        if token in stopwords.words('english'):
            clean_covid_tokens.remove(token)
    covid_freq = nltk.FreqDist(clean_covid_tokens)
    #for key,val in covid_freq.items():
        #print(str(key) + ':' + str(val))
    #covid_freq.plot(20, cumulative=False)
    wordcloud_covid = WordCloud().generate(covid_text)
    plt.imshow(wordcloud_covid, interpolation='bilinear')
    plt.axis("off")
    plt.show()

