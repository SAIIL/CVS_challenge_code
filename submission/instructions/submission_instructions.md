
Copyright &copy; SAGES. All Rights Reserved.

<div>
<a  href="https://cvs-challenge.grand-challenge.org/">
<img  src="https://rumc-gcorg-p-public.s3.amazonaws.com/b/652/CVS_Challenge_Media_-_Summit_Ad_3.x10.jpeg"  align="left"/>
</a>
</div>

# CVS Challenge Submission Instructions 
This document describes how to package your algorithm for submission to the SAGES Critical View of Safety Challenge. Other related info to challenge design, evaluation metrics, and more can be found on the [grandchallenge website](https://cvs-challenge.grand-challenge.org/) 

## Prerequisites
To be able to containerize and test your submission locally, you will need a GPU-equipped system with sufficient drivers installed. 
1. Install docker on your system to be able to containerize your model. Follow the instructions [here](https://docs.docker.com/engine/install/).
2. Install NVIDIA container toolkit to allow your containers to access your system GPU. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## Example code and structure
At this [repository](
https://github.com/SAIIL/CVS_challenge_code/tree/main/submission/example-algorithm), you will find some example code to help you 
1. Integrate your model.
2. Test your model locally.
3. Package your model to be submitted.


The code provided 
```
example-algorithm
│   inference.py
|   requirements.txt
|   save.sh
|   test_run.sh
|   Dockerfile
|
└───resources
│   │   constants.py
│   │   model.py
│   └───content
|   |      | some_resource.txt
|   |
|   └───utils
|   |      | file_ops.py
|
|
└───util
|   │   util.py
|
└───test
    └───input
    | 
    └───output

```

### Overview
The code above uses the **Dockerfile** to define what your final container should look like by these primary steps:
1. Defining a docker image to start from (e.g. from [these](https://hub.docker.com/r/pytorch/pytorch) precompiled pytorch containers).
2. Defining the required users and privileges needed to give us the right access to run your docker. Note that these should not be changed.
3. Copying your self-contained code repository located in the **resources** folder within the docker container.
4. Installing any additional packages needed to make your model run defined in **requirements.txt**.

### Integrating your algorithm
Within the **resources** folder you should be able to see what an example model looks like in model.py. Normally, you should be able to adapt your model along with any other required resources within this directory.

**Tip: Look out for the import structure defined for additional resources to be imported.**
  

### Testing locally

**inference.py** allows you to extract images given from a single input case (i.e. one 90-second long 1 fps mp4 video), call your model imported from the **resources** folder, and save the predictions. This code is only provided as an example but, once your model is set up in the resources folder, you should only have to change the code corresponding to your model call. This section of **inference.py** is explicitly marked.

Your docker image will only have access to a single input case at a time and you must strictly predict criteria corresponding to every frame. So, for one 90-second long 1 fps video you will have 90 (frames) x 3 (criteria) predictions made and saved as a json in the output folder. The code provided should help make sure that you follow the right output structure.

At evaluation time, for a single model run, your image will be provided access to the mounted input and output directories located at /input and /output within your docker container. 

**test_run.sh** allows you to build and test your container locally using an example case available at **./test/input**.

### Packaging your submission

**save.sh** allows you to save your built container as a tar.gz which can then be uploaded to the grandchallenge platform. 

Note that the saving process often takes several minutes but your saved container should appear after that.

Note that in this example process, the container building process happens alongside the testing using **test_run.sh**. While you can build your container separately, we highly recommend that you locally build and test any container that you plan to upload. We also recommend you build, test and save the example_model first to help familiarize yourself with this process.


### Submitting to grandchallenge
Note, to submit to the CVS challenge, at least one team member must have a grandchallenge account that is both:
1. Verified by grandchallenge. If you haven't already, you can verify your account [here](https://grand-challenge.org/verifications/create/).
2. Registered for the CVS challenge, please refer to the instructions [here](https://cvs-challenge.grand-challenge.org/instructions/).

This is the account you must use to finalize your submission.

#### Step 1 - Register your algorithm on grandchallenge

Once, you have packaged, tested, and saved your algorithm locally, you are now ready to upload it to the submission platform on grandchallenge. On the [CVS challenge page](https://cvs-challenge.grand-challenge.org/), navigate to [Submit tab](https://cvs-challenge.grand-challenge.org/evaluation/prelim-subchallenge-a-cvs-classification-submission/submissions/create/) and click on the link below the algorithm dropdown that will allow you to create a new algorithm on grandchallenge (see image below).

<div>
<a  href="https://cvs-challenge.grand-challenge.org/evaluation/prelim-subchallenge-a-cvs-classification-submission/submissions/create/">
<img  src="https://seafile.unistra.fr/f/8bd03b8ae00c4cd48492/?dl=1"  align="left"/>
</a>
</div>


#### Step 2 - Upload your algorithm to grandchallenge

Once you create an algorithm, you should automatically be taken to a page where you can upload your packaged container generated using **save.sh** by clicking on the "Upload a Container" button. Once your upload is complete and you fill in the corresponding details, hit save and wait for your container to become "Active". Note that this can take up to 20 minutes. 

You do not need to have your tab open during this process and you can always navigate back to your container on the dedicated page for your algorithm using the Algorithms tab on grandchallenge.

<div>
<a  href="https://grand-challenge.org/algorithms/">
<img  src="https://seafile.unistra.fr/f/bbd6abc629d84af78739/?dl=1"  align="left"/>
</a>
</div>

#### Step 3 - Test your container on grandchallenge

After your container becomes active, you can test it directly on grandchallenge by clicking on Try-out Algorithm. Note that this option only becomes visible on the page for your algorithm once you have an active container associated with that algorithm. You can drag and drop the same video provided alongside the example code at **./test/input/laparoscopic-video.mp4**. This is a 1 fps 90 second mp4 video that replicates testing conditions. You can download the resulting json and verify that it follows the same structure as your local test.

#### Step 4 - Submitting your algorithm

<div>
<a  href="https://cvs-challenge.grand-challenge.org/evaluation/prelim-subchallenge-a-cvs-classification-submission/submissions/create/">
<img  src="https://seafile.unistra.fr/f/d8f586501ef841f29da0/?dl=1"  align="left"/>
</a>
</div>

All active containers you uploaded then become available in the [Submit tab](https://cvs-challenge.grand-challenge.org/evaluation/prelim-subchallenge-a-cvs-classification-submission/submissions/create/) of the CVS challenge page. Submit your algorithm in the following order to submit to all the subchallenges using the various tabs shown in the image above.
1. Prelim Test Submission
2. Final Subchallenge A (CVS Classification) Submission
3. Final Subchallenge B (Uncertainty Quantification) Submission
4. Final Subchallenge C (Robustness) Submission


Note that you must wait for your submission to the Preliminary phase to be tested before you can submit to the subchallenges. You can check the status in the All Submissions tab.

Note that you must submit to all 3 subchallenges to be eligible for prizes.

# License
All data, videos, images, instructional material, etc. within this folder have been shared with the sole purpose of enabling participation for the CVS challenge.

  

# GOOD LUCK!