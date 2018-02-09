
# coding: utf-8
'''
Child Speech Patterns

I wanted to learn more about Natural Language Processing and apply it to
speech development in children. This is my first attempt at findings patterns
in children's speech at different ages. Eventually, constructing a model
for predicting speech delays based on age and speech patterns is my goal.

I found some transcripts of kids talking with their moms at childes.talkbank.org.

I chose to only look at the children's speech in this study, so the text is
made up of only the child's speech (not the mother's speech).
'''

# Importing Necessary Libraries
import numpy as np
import pandas as pd
pd.set_option('display.width', 2000)
from datetime import time
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import re
from collections import Counter
from nltk import word_tokenize, FreqDist, pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# Importing Transcripts From childes.talkbank.org
main_url = 'https://childes.talkbank.org/browser/index.php?url=Eng-NA/Bloom73/'

def grab_html(url):
    """
    Args:
        url (str) - website address
    Returns:
        soup (soup object)
    """

    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup

main_soup = grab_html(main_url)

# Retrieve the links to all transcripts
link_list = []
for a in main_soup.find_all('a', href = True)[4:]:
    link_list.append(a['href'])

print('There are %d links (transcripts).\n' % len(link_list))
print(link_list)
print('\nJust a second...')

# Get the html for each of the found transcripts
soup_list = []
for link in link_list:
    soup = grab_html(link)
    soup_list.append(soup)

print('\nAdded %d transcripts to our list.' % len(soup_list))




# Cleaning Up the Transcripts to Include Only The Child's Speech

# Filter html to just the speech parts
speech_list = []
for soup in soup_list:
    speech = soup.find('pre').text
    speech_list.append(speech)

# Filter speech to get just the child's speech
child_speech_list = []
speech_labels = []
for speech in speech_list:
    child_speech_list.append(re.findall('\*CHI:\t(.*)', speech))
    speech_labels.append(str(re.findall('CHI\|(.*?)\\|', speech)))

speech_labels = [re.sub('\[|\]|', '', label) for label in speech_labels]
speech_labels = [label.replace("'", '') for label in speech_labels]

print(speech_labels)

# remove the space and period after every utterance
child_speech_list = [[re.sub('\s\.', '', line) for line in speech]
                     for speech in child_speech_list]

# remove the comments made in square brackets (ie [grunting])
child_speech_list = [[re.sub('\[.*\]', '', line) for line in speech]
                     for speech in child_speech_list]

child_speech_list = [[re.sub('0', 'oh', line) for line in speech]
                     for speech in child_speech_list]
#print(child_speech_list[5])



# Create a DataFrame
child1_df = pd.DataFrame()
child1_df['age'] = speech_labels

# Number of Utterances within a Speech
child1_df['utterance_count'] = [len(speech) for speech in child_speech_list]

fig = plt.figure(figsize=(8, 7))
_ = sns.barplot(x = 'age', y = 'utterance_count', palette = sns.cubehelix_palette(8), data = child1_df)
_ = plt.title('\n\nNumber of Utterances Per Speech', size = 15)

# Total Word Count
def word_count(speech):
    """
    Args:
        speech (str or list object) - speech corpus
    Returns:
        word count per speech (float)
    """

    word_count = []

    for line in speech:
        count = len(line.split())
        word_count.append(count)

    return sum(word_count)

child1_df['word_count'] = [word_count(speech) for speech in child_speech_list]

fig = plt.figure(figsize=(8, 7))
_ = sns.barplot(x = 'age', y = 'word_count', palette = sns.cubehelix_palette(8), data = child1_df)
_ = plt.title('\n\nNumber of Words Per Speech', size = 15)
plt.show()

# Average Number of Words per Utterance
child1_df['avg_word/utterance'] = child1_df['word_count'] / child1_df['utterance_count']

fig = plt.figure(figsize=(8, 7))
_ = sns.barplot(x = 'age', y = 'avg_word/utterance', palette = sns.cubehelix_palette(8), data = child1_df)
_ = plt.title('\n\nAverage Number of Words Per Utterance', size = 15)
plt.show()


# Average Length of Words in Speech  -- NEED TO FINISH
# def avg_letters_word(speech):
#     """
#     Args:
#         speech (str or list object) - speech corpus
#     Returns:
#         average word length speech(float)
#     """
#
#     letter_count = []
#
#     for line in speech:
#         words = line.split()
#         for word in words:
#             letters = list(word)
#             num_letters = len(letters)
#             letter_count.append(num_letters)
#
#     return sum(letter_count)
# 
#
# child1_df['avg_letters_word'] = [avg_letters_word(speech) for speech in child_speech_list]
# child1_df

# Words Used Most Frequently
def top_10(speech):
    """
    Args:
        speech (str or list object) - speech corpus
    Returns:
        list of top 10 most frequent words (list)
    """

    words_list = []
    for line in speech:
        words = line.split()
        for word in words:
            words_list.append(word)

    freq_dist = FreqDist(words_list)
    most_freq = []
    for word in freq_dist.most_common(10):
        most_freq.append(word)
    print('Most Frequent Words:\n', most_freq, '\n')
    return most_freq

child1_df['top_10_words'] = [top_10(speech) for speech in child_speech_list]


# Parts of Speech
child1_df['%nouns'] = np.NaN
child1_df['%verbs'] = np.NaN
child1_df['%adverbs'] = np.NaN
child1_df['%adjectives'] = np.NaN
child1_df['%prepositions'] = np.NaN
child1_df['%pronouns'] = np.NaN
child1_df['%other'] = np.NaN

def pos_count(speech):
    """
    Args:
        speech (str or list object) - speech corpus
    Returns:
        A count of the number of parts of speech used (ie nouns, verbs) (int)
    """

    words_list = []
    for line in speech:
        words = line.split()
        for word in words:
            words_list.append(word)

    pos_list = []
    tagged = pos_tag(words_list)

    # Initalize POS Count
    nouns = 0
    verbs = 0
    adverbs = 0
    adjectives = 0
    prepositions = 0
    pronouns = 0
    other = 0
    total = 0

    # Count POS
    for word, tag in tagged:
        total += 1
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            nouns += 1
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            verbs += 1
        elif tag in ['RB', 'RBR', 'RBS', 'RP', 'WRB']:
            adverbs += 1
        elif tag in ['JJ', 'JJR', 'JJS']:
            adjectives += 1
        elif tag in ['IN']:
            prepositions += 1
        elif tag in ['PRP', 'PRP$', 'WP', 'WP$']:
            pronouns += 1
        else:
            other += 1

    # Record POS count as a percentage in dataframe
    child1_df.loc[i, '%nouns'] = nouns / total * 100
    child1_df.loc[i, '%verbs'] = verbs / total * 100
    child1_df.loc[i, '%adverbs'] = adverbs / total * 100
    child1_df.loc[i, '%adjectives'] = adjectives / total * 100
    child1_df.loc[i, '%prepositions'] = prepositions / total * 100
    child1_df.loc[i, '%pronouns'] = pronouns / total * 100
    child1_df.loc[i, '%other'] = other / total * 100

i = 0
for speech in child_speech_list:
    pos_count(speech)
    i += 1

pos_df = child1_df[['age', '%nouns', '%verbs', '%adverbs', '%adjectives',
                       '%prepositions', '%pronouns', '%other']]

pos_df = pos_df.set_index('age')
pos_df.columns = ['Nouns', 'Verbs', 'Adverbs', 'Adjectives', 'Prepositions',
                  'Pronouns', 'Other']

_ = pos_df.plot.area(figsize = [9, 8], colormap = 'Accent', title = 'Percent Parts of Speech Used at Each Age')
