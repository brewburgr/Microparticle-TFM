import urllib.request, tarfile, shutil

ddlab_link = 'http://micro.stanford.edu/~caiwei/Forum/2005-12-05-DDLab/ddlab-2007-12-18.tar.gz'
local_file = 'dd3d'

with urllib.request.urlopen(ddlab_link) as response:
    with open(local_file+'.tar.gz', 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
with tarfile.open(local_file+'.tar.gz') as tar:
    tar.extractall(path='.')
