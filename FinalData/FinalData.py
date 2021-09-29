# Libraries
from PIL import Image
from pathlib import Path
import numpy as np
from extract import extraction

if __name__ == '__main__':
    fe= extraction() #call extraction function and instiante it
    for img_path in sorted(Path("static/set").glob("*.jpg")):#set
        print(img_path)
        # Extract a deep feature here
        feature = fe.getInfo(img=Image.open(img_path)) 
        print(feature)
        #Obtain feature path
        feature_path = Path("static/feature11")/ (img_path.stem + ".npy") #feature1
        print(feature_path)

        # Save the features path to feature file
        np.save(feature_path,feature) 

