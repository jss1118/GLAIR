![glair](https://github.com/user-attachments/assets/4ee4f4cb-8ac9-44fe-9068-66ca70b83fe2)



# GLAIR | Predicting glioblastoma progression through temporal latent space variational autoencoders

<b>Get
involved: • [Issues](https://github.com/jss1118/Generative-VAE-Glioblastoma-Simulator/issues)</b>

## Abstract

Glioblastoma remains to be one of the deadliest tumors as of today, being diagnosed in 300,000+ people yearly. 
Optimized and specialized medical treatment has consistently proven to be one of the most effective ways of healing disease in patients, 
and is lackluster when it comes to carcinogenic diseases. 

Clinicians currently use a variety of methods for treating patients with brain tumors, 
however Chemotheray and radiation are most commonly used. While these methods of treatment have been proven tobe effective, they bring tolling 
side effects with them, which in some cases can contribute to surivval rates in patients. 

Having a clear path of the effectiveness of different treatment plans before intervening with the tumor can help patients recieve the most optimized care for their specific need. 
Many clinical trials have shown tumor growth remaining relatively unchanged in patients on a placebo drug, with no intervening with the cancerous tumor. In this project, I introduce a generative
variational autoencoder model, that works in the deep latent space to generate predictive simulations of tumor growth from CT1 segmented MRI scans, in response to either chemotherapy or 
surgical treatment in patients. (The model will soon also include survival rate/predictions, in order to help the patient decide whether it even makes sense for them to go through with chemotherapy.) 


## How does it work?

Creating simulations or predictions of future iterations of a vector usually requires a deep neural network. We initially tool the approach of developing a spatiotemporal transformer model, however this was prone to overfitting and was lackluster in feature extraction. 

We switched our approach to a generative variational autoencoder, which is better at recognizing important features. Variational autoencoders work by attempting to reconstruct an input image from a lower dimensional version of the input. We used this technique, but adjusted the loss function to reward the model when predicting the future progression rather than the input.
It takes a variety of input sequences from MRI slices, and then predicts a future vector from the input. 

![nn strcuture](https://github.com/user-attachments/assets/4b40de9c-ec3e-43a8-8de7-09354f7ffecd)


## Software/Model access

We are currently in a very early development stage, and are not available to open source our data yet. You can request access by contacting dearjoshuastanley@gmail.com

![eval_patient_3](https://github.com/user-attachments/assets/67302390-1af3-4048-8c8a-8322eafb322f)

<sup><i>The figure above represents 2 MRI timepoint slices, followed by the autoencoder prediction along with the ground truth</i></sup>

## Update cycle

**Alpha** These versions are not available to the public. They are early developer versions.

**Beta** These are early versions to updates that are available to the public for testing via request.

**Release** These are official updates that are available in the mainstream version.


## Contributing

You can contribute by contacting dearjoshuastanley@gmail.com, along with forking this repository.


## Copyright & license

This project is licensed under the **MIT License**








