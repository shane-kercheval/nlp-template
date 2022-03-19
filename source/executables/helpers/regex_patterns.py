"""
Most of these regex patterns are taken from: https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
"""
SUSPICIOUS = r'[&#<>:/{}\[\]\\]'
ANGLE_BRACKETS = r'<[^<>]*>'  # tags like <tab>
MARKDOWN_URLS = r'\[([^\[\]]*)\]\([^\(\)]*\)'  # markdown URLs like [Some text](https://....)
BRACKETS = r'\[[^\[\]]*\]'  # text or code in brackets like [0]
STANDALONE_SPECIAL_CHARACTERS = r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)'  # standalone sequences of specials, matches &# but not #cool
STANDALONE_SEQUENCES = r'(?:^|\s)[\-=\+]{2,}(?:\s|$)'  # standalone sequences of hyphens like --- or ==
WHITESPACE = r'\s+'  # sequences of white spaces
