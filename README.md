# TMLS_Operational_AI_Workshop
A repo with assets required for the TMLS operational AI virtual workshop

# Login to Continual using the emailed invite link 
If you didn’t receive an invite link, please raise a hand and let Brendan McKenna know. 


After logging in, you should see a project named **Topic_ClassificationX** connected to a demo Snowflake database and a classification model already created. The demo Snowflake database includes a dataset of consumer complaints filed with the Bureau of Consumer Financial Protection.

Connecting a Continual project to a cloud data warehouse like Snowflake, BigQuery, Redshift, or Databricks is easy, simply [choose a vendor and enter your credentials](https://docs.continual.ai/datastore-overview/?h=connecting#data-warehouses).

# (Optional) Create your own project
If you are stepping through the workshop on your own, want to create your own project, or didn't received an invite to a project, register for a trial at https://cloud.continual.ai/register and create a new project. 

# (Optional) Connect to a cloud data warehouse
If you were added to a workshop project titled Topic_Classification then you can skip this step.[Connect to your cloud data warehouse](https://docs.continual.ai/datastore-overview/?h=connecting#data-warehouses) and create a table from the consumer_complaints csv in the workshop [github repository](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop). 


# Install Continual CLI
Install the Continual CLI into a virtual environment. Run the following bash commands: 

`virtualenv operational_ai_workshop`
`cd /operational_ai_workshop && . bin/activate`
`pip3 install continual`

After the installation is complete, login using your Continual credentials and select the **Topic_Classification** project you were added to or created. 

# Clone Workshop Git Repository
Clone this workshop github repo. 

`git clone https://github.com/b-mckenna/TMLS_Operational_AI_Workshop`

The repo consists of a model definition and custom algorithm extension. The model definition includes attributes such as name, primary index, and a data table. Model definitions can contain ids to other data entities which Continual joins with the model to build the training set. We typically recommend storing your features in feature sets and connecting your models to them via entity linking, but it's also possible to specify individual columns as input features to your model as we have done in this simple example. 

Along with the core model definition, the model YAML defines training, promotion, and prediction policies for Continual to execute. In our example, we define a set of algorithms to run in the experiment to ensure our predictions are generated from the best performing model according to our evaluation metric. Other algorithm configurations such as hyperparameters can easily be declared. Continual has a default set of algorithms per problem type, and users can add custom algorithms. See the YAML reference in our documentation for more options. 

To add a custom algorithm, users can run continual extensions init –type algorithm –id MyCustomAlgorithm and write Python code to implement an algorithm of their choosing. In this workshop, we use distilbert, Tensorflow, and Huggingface to create a custom algorithm to include in our model experiment. All of our predictions are written back to Snowflake for downstream applications to use. 
# Edit your model definition
In your favorite editor, open and edit /models/product_classification.yaml. 
Change the name field from product_classification_example to product_classification_firstname_lastname

*NOTE: If you created your own project, you need to run `continual push extensions/extension.yaml` before you push the model yaml in the next step.* 

Once you’ve edited and saved product_classification.yaml, submit it to Continual from the CLI using the push command. 

`continual push ./model/product_classification.yaml`

*NOTE: You don't need to change the file name, only the name field in the file needs to be changed to avoid conflicts.*

Copy and paste the link to cloud.continual.com into your browser that was generated from the ‘continual push’ command. 

# Reviewing Performance
The link will take you to the Changes page. Continual Changes are a complete record of the changes that have occurred in the project. If you click into a change you can monitor the progress. 

To accomodate the time constraint of this abbreviated workshop, we used a 100 row sample of the dataset and only 4 algorithms. Consequently, performance will be subpar but that's okay. The purpose of the workshop is to exemplify the workflow. Improving model performance won't be covered here. 

If you'd like to train on a slightly larger dataset, edit your model yaml to table `tmls.cfpb.sample_consumer_complaints`. It will take roughly 1h45m to complete training for each algorithm. 

*NOTE: We did not enable GPUs for this workshop. All training is being done on CPUs.* 

Once the training has completed and you are in the Model Version page, review the performance of each algorithm across the evaluation metrics and train, validation, test, and evaluation datasets. 

Click on Model Insights to analyze distilbert’s performance by reviewing the Local interpretable model-agnostic explanations (LIME). 

These charts, along with the models, are downloadable from the Artifacts tab. 

Return to the Model page and scroll to the bottom to review the Snowflake tables housing our predictions. 

# Conclusion
Our model is live and being continually maintained and monitored. Predictions will be refreshed weekly and the performance distribution is tracked over time. At any point, you can add another algorithm, a custom SOTA or default, to the experiment. 
