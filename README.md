# Smartcrop
Smartcrop is a multi-pass context-aware cropping tool

Usage:
> Make sure source images are in the source folder.
```
python smartcrop.py -i "[filename.jpg]" -x 300 - y 300
```

Will take filename.jpg and produce reference information images and the final crop in /output

![original] [org]
![edges] [edges]
![blocks] [blocks]
![final image] [final]

> Usable mage information changes depending on crop resolution
```
python smartcrop.py -i "[filename.jpg]" -x 400 - y 300
```

![different] [final2]

[org]: source/debate.jpg
[edges]: doc/edges.jpg
[blocks]: doc/blocks.png
[final]: doc/cropped_debate.jpg
[final2]: doc/cropped_debate2.jpg