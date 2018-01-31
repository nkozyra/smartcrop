# Smartcrop
Smartcrop is a multi-pass context-aware cropping tool

Usage:
> Make sure source images are in the source folder.
```
python smartcrop.py -i "[filename.jpg]" -x 300 - y 300
```

Will take filename.jpg and produce reference information images and the final crop in /output

![original](https://raw.githubusercontent.com/nkozyra/smartcrop/master/source/debate.jpg)
![edges](https://raw.githubusercontent.com/nkozyra/smartcrop/master/doc/edges.jpg)
![blocks](https://raw.githubusercontent.com/nkozyra/smartcrop/master/doc/blocks.png)
![final image](https://raw.githubusercontent.com/nkozyra/smartcrop/master/doc/cropped_debate.jpg)

> Usable image information changes depending on crop resolution
```
python smartcrop.py -i "[filename.jpg]" -x 400 - y 300
```

![different](https://raw.githubusercontent.com/nkozyra/smartcrop/master/doc/cropped_debate2.jpg)
