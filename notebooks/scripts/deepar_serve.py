import logging
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from io import BytesIO, StringIO
import pandas as pd
import os
import json
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tarfile
model_dir = '/opt/ml/model'

def change_permissions(directory_path, permissions):
    """
    Change permissions for all files and directories within the specified directory.

    Args:
        directory_path (str): Path to the directory.
        permissions (int): Octal value representing the desired permissions.
    """
    try:
        for root, dirs, files in os.walk(directory_path):
            # Change permissions for files
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.chmod(file_path, permissions)
                except Exception as e:
                    print(f"Error changing permissions of {file_path}: {e}")
            # Change permissions for directories
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    os.chmod(dir_path, permissions)
                except Exception as e:
                    print(f"Error changing permissions of {dir_path}: {e}")
        print("Permissions changed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def print_directory_structure(directory):
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        logger.info('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            logger.info('{}{}'.format(subindent, file))

def model_fn(model_dir):
    """Loads model from previously saved artifact"""
    logger.info("Loading model from directory: %s", model_dir)
    model = TimeSeriesPredictor.load(model_dir, require_version_match=False)
    # directory_path = '/opt/ml/model/models/'
    # permissions = 0o777  
    # change_permissions(directory_path, permissions)
    #logger.info(f"Model stats:{model},kcv names:{model.known_covariates_names},pred length: {model.prediction_length},eval metric: {model.eval_metric}")
    logger.info("Model loaded successfully...")
    return model

def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        logger.warning("More than one file is found in %s directory", path)
    logger.info("Using file: %s", file)
    filename = f"{path}/{file}"
    return filename

def get_last_n_rows(group, n):
    return group.tail(n)

def get_test_related_ts(test_data, prediction_length, related_ts):
    
    known_covariates = test_data.groupby(level="item_id").apply(get_last_n_rows, n=prediction_length)[related_ts]
    
    # Reset the multi-index for cleaner output
    known_covariates = known_covariates.reset_index(level=0, drop=True)
    return known_covariates

def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
        logger.info("%s found in environment variables with value: %s", name, result)
    else:
        logger.warning("%s not found in environment variables", name)
    return result

def transform_fn(
    model, request_body, input_content_type, output_content_type="text/csv"
):
    logger.info("Transform function called with input_content_type: %s", input_content_type)
     
    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf,names=['item_id','timestamp']+ model.known_covariates_names)
        logger.info(f"Data reading successful. Cols are...{data.columns}")

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = pd.read_json(buf)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = pd.read_json(buf, orient="records", lines=True)

    else:
        raise ValueError(f"{input_content_type} input content type not supported.")
    logger.info("converting to ts df...")
    known_covariates = TimeSeriesDataFrame.from_data_frame(data)
    logger.info("converted to Timeseries DF")
    #logger.info(f"contents of the directory /opt/ml/model/code:{os.listdir('/opt/ml/model/code')}")
    #print_directory_structure('/opt/ml')
    #config_path = get_env_if_present("CONFIG_FILE_URI") 
    #config_file = get_input_path(config_path)
    #logger.info(f"Config file path is {config_file}")
    #with open(config_file) as f:
    #    config = yaml.safe_load(f)  # AutoGluon-specific config
    
    # ag_predictor_args = json.loads(get_env_if_present("ag_predictor_args"))
    # test_end_date = get_env_if_present("test_end_date")
    # prediction_length = ag_predictor_args['prediction_length']
    # test_start_date = pd.Timestamp(test_end_date) - pd.Timedelta(days=prediction_length)
    # ag_predictor_args['known_covariates_names'] = ag_predictor_args['known_covariates_names'].split(',')
    # logger.info(f"args for prediction are {ag_predictor_args}")
    # known_covariates = get_test_related_ts(test_data=test_data[test_data.index.get_level_values("timestamp")<=test_end_date], 
    #                                        prediction_length=prediction_length, 
    #                                        related_ts= ag_predictor_args['known_covariates_names']
    #                                       )
    #train_data = test_data[test_data.index.get_level_values("timestamp")<test_start_date]
 
    train_data = TimeSeriesDataFrame.from_pickle(f'{model_dir}/utils/data/train.pkl')
    kvc_item_list = known_covariates.index.get_level_values('item_id').unique().tolist()
    train_data = train_data.loc[kvc_item_list]
    logger.info(f"train data loaded.. cols are {train_data.columns}")
    logger.info(f"known covariates loaded.. cols are {known_covariates.columns}")
    try:
        prediction = model.predict(train_data, known_covariates=known_covariates,random_seed=123)
    except Exception as e:
        error_message = f"caught exception {e}"
        logger.error(error_message)
        # Return error message as CSV-formatted string
        return error_message +",1,\n1", "text/csv"
    logger.info('Predictions from model returned')
    
    
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
        output_content_type = "application/x-parquet"
    elif "application/json" in output_content_type:
        output = prediction.to_json()
        output_content_type = "application/json"
    elif "text/csv" in output_content_type:
        output = prediction.reset_index().to_csv(index=None)
        output_content_type = "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")
    logger.info("post processing done.. returning")
    return output, output_content_type
