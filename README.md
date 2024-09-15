# TMLS_Operational_AI_Workshop
This Github repo contains all the assets needed to complete the [TLMS Operational AI virtual workshop](https://hopin.com/events/toronto-machine-learning-micro-summit-on-nlp/registration?code=MphOkEwsKmzZit8hwoGMO1q2n) delivered by [Continual](https://continual.ai/). 

# Logging in to Continual using the invite link 
For those who pre-registered to the workshop, you should have received an invite in your inbox. Click on the link and sign up for a free trial of Continual! If you didn't receive an invite and would still like to participate, please skip to the next step (Creating your own project). 

After logging in, you should see a project named **Topic_Classification** connected to a demo Snowflake database and a classification model already created. The demo Snowflake database includes a dataset of consumer complaints filed with the Bureau of Consumer Financial Protection.

# (Optional) Creating your own project
If you'd like to step through the workshop yourself, create your own project, or didn't receive an invite to the **Topic_Classification** project, you can register for a trial directly by navigating [here](https://cloud.continual.ai/register) and creating your own project.  

# (Optional) Connecting to a cloud data warehouse
If you have been added to the aforementioned **Topic_Classification** project, then please skip the rest of this step. 

Otherwise, connecting your own Continual project to a cloud data warehouse like Snowflake, BigQuery, Redshift, or Databricks is an easy process. Simply [choose your vendor](https://docs.continual.ai/datastore-overview/?h=connecting#data-warehouses), enter the appropriate credentials, and then any remaining connection parameters that might be required. 

If you are creating your own project, you will also want to load one or both of the *consumer_complaints* datasets provided from our workshop github repository [here](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop/tree/main/data) into a table in your database. 

## (Optional) Loading your data into Snowflake

Assuming that you are creating your own project and following along, to help streamline the process for you, we have provided the following [ddl.sql](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop/blob/main/sql/ddl.sql) script that you can use to create the table definition in your target Snowflake schema and then a [snowsql_staging.sql](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop/blob/main/sql/snowsql_staging.sql) script to load the data through `snowsql`.

Follow these steps:
1. Download the *micro_consumer_complaints.csv* and *sample_consumer_complaints.csv* files into your local directory by running the following commands:
```sh
wget https://raw.githubusercontent.com/b-mckenna/TMLS_Operational_AI_Workshop/main/data/micro_consumer_complaints.csv
wget https://raw.githubusercontent.com/b-mckenna/TMLS_Operational_AI_Workshop/main/data/sample_consumer_complaints.csv
```
2. Make sure that you have installed [SnowSQL](https://docs.snowflake.com/en/user-guide/snowsql-install-config.html)
3. Copy the SQL from the [ddl.sql](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop/blob/main/sql/ddl.sql) script into a worksheet in the Snowflake Web UI and execute the statements (updating the database name, schema name, and role accordingly)
3. Using `snowsql`, execute the provided [snowsql_staging.sql](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop/blob/main/sql/snowsql_staging.sql) script [through the CLI](https://docs.snowflake.com/en/user-guide/snowsql-use.html#running-batch-scripts)
```sh
snowsql -a <account_name> -u <username> -w <WAREHOUSE_NAME> -f ./snowsql_staging.sql
```
> ℹ️ For the **snowsql_staging.sql** script, please update the script to point to the fully qualified paths of your local *consumer_complaints.csv* files. You will also want to update the database name and schema if they are different from "TMLS" and "CFPB" respectively. 

> ℹ️ Your database name and schema name should ideally match what you specified when creating the connection for your project. If you left the schema blank when defining the connection for your project, the schema will be the same as your project ID (look in the URL).

# Installing the Continual CLI
[Follow our documentation](https://docs.continual.ai/installing/) to install the Continual CLI into a virtual environment. 

Alternatively, run the following bash commands:

```sh
virtualenv operational_ai_workshop
cd operational_ai_workshop/ && . bin/activate
pip3 install continual
```

> ℹ️ Make sure you have python 3.8 or newer installed on your local machine

After the installation is complete, you can [login to Continual](https://docs.continual.ai/installing/#logging-in) using your credentials and select the **Topic_Classification** project (or your own project if doing it yourself).

# Cloning the Workshop Git Repository
Clone this github repo to proceed with the workshop. 

```sh
git clone https://github.com/b-mckenna/TMLS_Operational_AI_Workshop
```

The repo consists of a model definition and a custom algorithm extension. The model definition, stored as a [model YAML file](https://docs.continual.ai/yaml-reference/#model-yaml-reference), contains attributes, such as the name of the model, the index being used, and a reference to the data table that contains the [model spine](https://docs.continual.ai/models/#model). Model definitions can contain `ids` to other data [entities](https://docs.continual.ai/feature-sets/#entity), which Continual will then join with the model to build the training set. We typically recommend storing your features in [feature sets](https://docs.continual.ai/feature-sets/#feature-set) for easier re-use and connecting your models to them via entity linking, but it's also possible to specify individual columns as input features to your model directly as we have done in this example. 

Along with the core model definition, the model YAML defines the training, promotion, and prediction policies that Continual will be using with the model. In our example, we define a set of algorithms to run in the experiment to ensure our predictions are generated from the best performing model according to our evaluation metric. Other algorithm configurations, such as hyperparameters, can easily be declared as well. For each problem type, Continual has a default set of algorithms that it uses but users can specify additional custom algorithms to be added. For more details and options, please refer to the [YAML reference page](https://docs.continual.ai/yaml-reference/) in our documentation. 

Finally, to add a custom algorithm, users can write custom Python code to implement an algorithm of their choosing and run the following command:

```sh
continual extensions init –type algorithm –id MyCustomAlgorithm
```
> ℹ️ We have provided an extension with this workshop already so you don't need to create your own

In this particular workshop, we use [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert), [Tensorflow](https://www.tensorflow.org/), and [Huggingface](https://huggingface.co/) to create a custom algorithm to include in our model experiment. All of our predictions are then written back to Snowflake for downstream applications to use / consume.

## (Optional) Initializing your extension

If you created your own project, you will first need to run the following command before pushing your model YAML file in the following steps:
```sh
continual push extensions/distilbert/extension.yaml
```
> ⚠️ If you are using a different python version locally than python 3.8 (e.g. python 3.9 or 3.10), you may need to edit the python version under `/extensions/distilbert/pyproject.toml` > `tool.poetry.dependencies` > `python` to match your local python version. Please feel free to reach out to support@continual.ai if you need help with activating the extensions feature in your project.

If you are following along with the workshop and using the pre-existing **Topic_Classification** project, feel free to skip this step. 

# Editing your model definition
In your favorite text editor, open and edit the `/models/product_classification.yaml` file. Change the `name` field from *product_classification_example* to *product_classification_<firstname>_<lastname>*.

Once you've edited and saved [product_classification.yaml](https://github.com/b-mckenna/TMLS_Operational_AI_Workshop/blob/main/models/product_classification.yaml), you can then submit it to Continual through the CLI by using the push command:

```sh
continual push ./models/product_classification.yaml
```
> ℹ️ You don't need to change the file name itself. Notably, only the "name" field within the file needs to be changed to avoid any conflicts. 

Afterwards, copy and paste the resulting URL link that was generated from the `continual push` command into your browser to view your progress. 

# Reviewing Performance
This URL link will take you to the `Changes` tab of your project, which contains all historical changes that have been made in your project. If you click into a change, you can monitor the progress.  

To accomodate the time constraint of this abbreviated workshop, we've used a 100 row sample of the dataset and only 4 algorithms. Therefore, performance will be suboptimal and only meant for illustrative purposes. We will not be covering tuning model performance in this workshop. 

If you'd like to train on a slightly larger and more representative dataset, you can edit your `product_classification.yaml` file to point to the table `tmls.cfpb.sample_consumer_complaints` instead. In this case, it will take roughly 2 hours to complete full model training, promotion, analysis, and deployment. 

> ℹ️ If you have created your own project and loaded the data into your own cloud data warehouse, update the `table` field to point to the appropriate fully qualified table name instead (whether it's *micro_consumer_complaints* or *sample_consumer_complaints*).

> ℹ️ For this workshop, we did not enable GPUs and all training will be performed on CPUs.

Once the model training has finished and you are in the `Model Version` page, you can review the performance of each algorithm across the evaluation metrics and train, validation, test, and evaluation datasets. 

Click on `Model Insights` to analyze DistilBERT’s performance by reviewing the Local Interpretable Model-Agnostic Explanations (LIME). These charts, along with the models, are downloadable from the `Artifacts` tab. 

Navigate back to the `Model Overview` and scroll to the bottom to review the Snowflake tables housing our predictions. 

# Conclusion
Congratulations, our model is now live and being continually maintained / monitored! 

Going forward, predictions will be refreshed on a weekly basis and the performance distribution will be tracked over time. At any point, you can easily choose to add another algorithm, a custom SOTA or default, to the experiment based on your needs.
