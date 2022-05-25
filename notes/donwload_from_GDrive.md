##### [gdown](https://github.com/wkentaro/gdown)

Download a large file from Google Drive.

#### how to use?

1、in shell

```shell
gdown gdrive_link
```

2、in python

```python
import gdown
url = 'link'
output = 'file_name'
# download
gdown.download(url, output, quiet=False)
# checkout
md5 = 'md5_code'
gdown.cached_download(url, output, md5=md5, postprecess=gdown.extractall)
```

