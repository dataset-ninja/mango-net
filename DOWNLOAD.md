Dataset **MangoNet** can be downloaded in Supervisely format:

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/O/C/Xb/yVTE1R1NMZTA3ufUU02NDqf9zBKY0q5iA7F7VrjQh8HAXbJIv17ium5oKtjLjoXstWEP9aN5PHpUnu5b7mYBdQEMQ6mraEp4RwmjcqYEBtSCusP9HHbrep7Jqylp.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='MangoNet', dst_path='~/dtools/datasets/MangoNet.tar')
```
The data in original format can be 🔗[downloaded here](https://github.com/dataset-ninja/mango-net)