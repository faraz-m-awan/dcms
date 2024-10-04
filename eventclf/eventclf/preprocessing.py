# import cleaning_process
from html.parser import HTMLParser
from nltk.stem.wordnet import WordNetLemmatizer
import re
import calendar

from nltk.corpus import stopwords
stop_words = stopwords.words("english")
# stop_words = []

hash_tags = ['creamfields2016', 'creamfields', 'creamfield'] + ['vfestival', 'V21st', 'vfest2016', 'vfest', 'v festival', 'v fest']

html_parser = HTMLParser()
lemmatizer = WordNetLemmatizer()
time_re = re.compile(r'^(([01]\d|2[0-3]):([0-5]\d)|24:00)$')
months = [calendar.month_name[i].lower() for i in range(1, 13)]
months.append('aug')

urlre = re.compile(r'http[s]{,1}://[^ ]+')
mentionre = re.compile(r'@[\w]+')
questionre = re.compile(r'\?+')
soore = re.compile(r'soo+')
hashre = re.compile(r'#[\w]+')
emotre = re.compile(
    r'(:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')
featre = re.compile(
    r'(http[s]{,1}://[^ ]+|[\w\-]+|#\w+|@\w+|\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[;:,\!\.\?]|$)')


def is_time_format(s):
    return bool(time_re.match(s))


def represents_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def clean_html(html):
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    cleaned = re.sub(r"(?s)<[/\w].*?>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    return cleaned.strip()


def ngrams(items, n, prefix):
    return [prefix + '_'.join(items[start:start + n]) for start in
            range(0, len(items) - n + 1)]


def rich_analyzer_textual(doc):
    word_ngrams = [2, 3]
    stopwords = set(stop_words)
    doc = clean_html(doc)

    doc = doc.encode('utf-8')
    doc = str(doc).lower()

    for tag in hash_tags:
        doc = str(doc).replace("#" + tag, '#festivalname')
        doc = str(doc).replace("@" + tag, '@festivalname')
        doc = str(doc).replace(tag, 'festivalname')
        doc = str(doc).replace('£', ' pounds ')

    output = list()
    questions = ['?(question_mark)' for x in questionre.findall(doc)]
    output.extend(featre.findall(doc))

    output = map(lambda x: lemmatizer.lemmatize(x, 'v'), output)
    output = map(lambda x: lemmatizer.lemmatize(x, 'a'), output)

    output = map(lambda x: x if not represents_int(x) else 'numberint', output)
    output = map(lambda x: x if not represents_float(x) else 'numberfloat',
                 output)
    output = map(lambda x: x if not is_time_format(
        str(x).replace('pm', '').replace('am', '')) else 'hourdate', output)
    output = map(lambda x: x if not x in months else 'monthdate', output)
    output = map(lambda x: 'you' if str(x) in ['u'] else x, output)
    output = map(lambda x: 'jealous' if str(x) in ['jel'] else x, output)
    output = map(lambda x: 'not' if str(x) in ['not', "n't", 'no'] else x,
                 output)
    output = [x if len(soore.findall(x)) == 0 else 'so' for x in output]

    if word_ngrams is None:
        word_ngrams = list()
    ngm = list()
    for n in word_ngrams:
        ngm.extend(ngrams(output, n, '_W%iG_' % n))
    output.extend(ngm)

    output = [x for x in output if len(x) > 1 and not x in stopwords]
    output = [x for x in output if
              len(x) > 1 and not x in ['#festivalname', '@festivalname',
                                       'festivalname']]
    output.extend(questions)

    return output


def rich_analyzer_w2v(doc):
    stopwords = set(stop_words)

    doc = clean_html(doc)
    doc = str(doc).lower()

    for tag in hash_tags:
        doc = str(doc).replace("#" + tag, '')
        doc = str(doc).replace("@" + tag, '')
        doc = str(doc).replace(tag, '')

    output = list()

    questions = ['question' for x in questionre.findall(doc)]
    output.extend(featre.findall(doc))
    output.extend(questions)

    output = map(lambda x: x if not represents_int(x) else 'number', output)
    output = map(lambda x: x if not represents_float(x) else 'number', output)
    output = map(lambda x: x if not is_time_format(
        str(x).replace('pm', '').replace('am', '')) else 'hour', output)
    output = map(lambda x: 'you' if str(x) in ['u'] else x, output)
    output = map(lambda x: 'jealous' if str(x) in ['jel'] else x, output)
    output = map(lambda x: 'not' if str(x) in ['not', "n't", 'no'] else x,
                 output)

    output = [x for x in output if len(x) > 0 and not x in stopwords]
    output = [x for x in output if len(urlre.findall(x)) == 0]
    output = [x if len(soore.findall(x)) == 0 else 'so' for x in output]
    output = [x if len(hashre.findall(x)) == 0 else x.replace('#', '') for x in
              output]
    output = [x if len(mentionre.findall(x)) == 0 else x.replace('@', '') for x
              in output]

    return output
