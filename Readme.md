Order of setting up project:

Step1:- 
1) install required libraries using command: "pip install -r requirements.txt"
2) dowload dataset, make a folder dataset and make three subfolders: Bot-IoT,UNSW-NB15,cleaned_dataset
3) download link: usnw_dataset_link:https://cloudstor.aarnet.edu.au/plus/apps/onlyoffice/s/2DhnLGDdEECo4ys?fileId=206777051 and https://cloudstor.aarnet.edu.au/plus/apps/onlyoffice/s/2DhnLGDdEECo4ys?fileId=206779316 and put it inside dataset/UNSW-NB15
4) download bot_iot data_set: https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE?path=%2FCSV%2FTraning%20and%20Testing%20Tets%20(5%25%20of%20the%20entier%20dataset)%2FAll%20features and put it inside dataset/Bot-Iot


Step2:-
1) Copy the cleaned_dataset  folder into normal_ddos_dos_classification
2) Run data_cleaning_unsw.ipynb and data_cleaning_bot_iot.ipynb, to clean dataset
3) Run the training.ipynb, to train the model
4) Run the test.ipynb, to test  the model

Common features used:

Spkts in bot ~ Spkts in unsw  (Source-to-destination packet count)
Dpkts in bot ~ Dpkts in unsw  (Destination-to-source packet count)
Sbytes in bot ~ sbytes in unsw  (Source-to-destination byte count)
Dbytes in bot ~ dbytes in unsw  (Destination-to-source byte count)
Dur in bot ~ dur in unsw (record total duration)
Proto in bot ~ proto in unsw (transaction protocol)
attack in bot ~  label in unsw (traffic: 0=normal traffic,1=attack traffic)
stime in bot ~ Stime in unsw  (record start time)
ltime in bot ~ ltime in unsw  (record last time)
sport in bot ~ sport in unsw  (source port number)
dport in iot ~ dsport in unsw (destination port number)
saddr in iot ~ srcip in unsw  (source ip address)
daddr in iot ~ dstip in unsw  (destination ip address)

Ouput label:
category in bot: (Dos/Ddos)
category in unsw: (normal)