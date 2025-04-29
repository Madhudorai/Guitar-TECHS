# Guitar-TECHS
Collab notebook for loading Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords, and Scales. 
https://colab.research.google.com/drive/1ykem3MaFS2XqVXz7ZniUxtqXQL0vqpK4?usp=sharing
# About Dataset

![image](https://github.com/user-attachments/assets/050de7d4-f74d-4d23-8974-d259663f9dbc)

For more informationn about the dataset visit https://guitar-techs.github.io/

# Collab Notebook 
Open the .ipynb in google colab. It automatically downloads the dataset from Zenodo URL and uses pytorch dataloaders to load the dataset. You can filter the dataset through 
a) players= 'P1','P2','P3' and combinations of these
b) content types = 'scales', 'chords','music', 'techniques', 'singlenotes' 
c) modalities = 'directinput', 'micamp', 'exo', 'ego'

Labels of note, note onset, note offset, fret, string are generated using the MIDI file, we can decide the time slice of the sample (using slice range or slice size)- so that all modalities and the corresponding labels are aligned for each sample. 
We can hear the different modalities and look at the labels for each sample. 

# Citations: 
Pedroza, Hegel, et al. "Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and Scales Using a Diverse Array of Hardware." ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025
