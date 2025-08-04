# chord-recognition
A repo for chord recognition

## Models
### CR1
 - A CRNN encoder model based on CR1 model from [STRUCTURED TRAINING FOR LARGE-VOCABULARY CHORD
RECOGNITION](https://brianmcfee.net/papers/ismir2017_chord.pdf)

#### Model specific configurations
- dropout = 3 value array
- hidden = 1 value array

### SimpleLSTM
 - A simple lstm model mainly used in conjunction with extracted chroma features

#### Model specific configurations
- dropout = 1 value array
- hidden = 1 value array