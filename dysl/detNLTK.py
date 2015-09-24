#!/usr/bin/env python
#coding:utf-8
# Inspired in Alejandro Nolla - Example for detecting language using a stopwords based approach
# Updated by ClarissaCastella

import sys

try:
    from nltk import wordpunct_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print '[!] You need to install nltk (http://nltk.org/index.html)'



#----------------------------------------------------------------------
#Requires stopwords files: /home/USER/nltk_data/corpora/stopwords
class detLNTK:
    	@classmethod

	def _calculate_languages_ratios(self,text):
	    """
	    Calculate probability of given text to be written in several languages and
	    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}
	    
	    @param text: Text whose language want to be detected
	    @type text: str
	    
	    @return: Dictionary with languages and unique stopwords seen in analyzed text
	    @rtype: dict
	    """

	    languages_ratios = {}

	    '''
	    nltk.wordpunct_tokenize() splits all punctuations into separate tokens
	    
	    >>> wordpunct_tokenize("That's thirty minutes away. I'll be there in ten.")
	    ['That', "'", 's', 'thirty', 'minutes', 'away', '.', 'I', "'", 'll', 'be', 'there', 'in', 'ten', '.']
	    '''

	    tokens = wordpunct_tokenize(text)
	    words = [word.lower() for word in tokens]

	    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
	    for language in stopwords.fileids():
		#print language
		stopwords_set = set(stopwords.words(language))
		words_set = set(words)
		common_elements = words_set.intersection(stopwords_set)

		languages_ratios[language] = len(common_elements) # language "score"

	    return languages_ratios


	#----------------------------------------------------------------------
	def detect_language(self,text):
	    """
	    Calculate probability of given text to be written in several languages and
	    return the highest scored.
	    
	    It uses a stopwords based approach, counting how many unique stopwords
	    are seen in analyzed text.
	    
	    @param text: Text whose language want to be detected
	    @type text: str
	    
	    @return: Most scored language guessed
	    @rtype: str
	    """

	    ratios = self._calculate_languages_ratios(text)

	    most_rated_language = max(ratios, key=ratios.get)

	    return most_rated_language




