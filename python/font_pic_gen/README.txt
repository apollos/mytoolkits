#Usage
usage: font_pic_gen.py [-h] [--bg-img-path BACKGROUND] --fonts-path FONTS
                       [--config-file CONFIGURATION] --characters-path
                       CHARACTERS [--target-num TARGET] [--output-path OUTPUT]
                       [--font-size-list {SMALL,MED,LARGE} [{SMALL,MED,LARGE} ...]]
                       [--seed SEED] [--char-num CHAR_NUM]
                       [--white-font [WHITE_FONT]]

optional arguments:
  -h, --help            show this help message and exit
  --bg-img-path BACKGROUND
                        Path to folders of background images. If not set, will
                        use black and white as background
  --fonts-path FONTS    Path to folder of fonts
  --config-file CONFIGURATION
                        Path to the json configuration file for image
                        generation
  --characters-path CHARACTERS
                        Path to the candidate character files
  --target-num TARGET   To be generated image number
  --output-path OUTPUT  Output image location
  --font-size-list {SMALL,MED,LARGE} [{SMALL,MED,LARGE} ...]
                        To be used font size. It can be multiple choices
  --seed SEED           Random Seed
  --char-num CHAR_NUM   Generated character number. If not set, will be random
                        between 3 - 20
  --white-font [WHITE_FONT]
                        Add white color font in the generate list


#CMD Example:
##With Background Image
python font_pic_gen.py --fonts-path fonts --characters-path words --target-num 100 --bg-img-path background/

##Without Background Image and use gnerated Background color and use white color font
python font_pic_gen.py --fonts-path fonts --characters-path words --target-num 100 --white-font
