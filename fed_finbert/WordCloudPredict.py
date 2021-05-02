from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import os
import string
import random

class WordCloudPredict:
    def __init__(self, temp_path):
        self.wordcloud = WordCloud(background_color="white", max_words=1000)
        self.temp_path = temp_path

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        
        return ''.join(random.choice(letters) for i in range(length))

    def predict(self, s_doc):
        # Generate a word cloud image
        cloudimage = self.wordcloud.generate(s_doc)

        # Display the generated image
        fig = plt.figure(figsize = (12, 8), facecolor = None) 
        plt.imshow(cloudimage, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad = 0) 
        plt.show()

        temp_file_name = self.temp_path + self.get_random_string(10) + '.png'

        fig.savefig(temp_file_name)

        with open(temp_file_name, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        os.remove(temp_file_name)

        return encoded_string
