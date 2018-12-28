import re
ex = '2012/01/03 01:00:00.000'
phoneNumRegex = re.compile(r'\d\d:\d\d\:\d\d.\d\d\d')
mo = phoneNumRegex.search(ex)
print mo.group()