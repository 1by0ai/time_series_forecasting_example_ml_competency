import logging
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from io import BytesIO, StringIO
import pandas as pd
import os
import json
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_dir = "/opt/ml/model"
def model_fn(model_dir):
    """Loads model from previously saved artifact"""
    logger.info("Loading model from directory: %s", model_dir)
    model = TimeSeriesPredictor.load(model_dir, require_version_match=False)
    logger.info("Model loaded successfully...")
    return model

def unpack_data(packed_df):
    unpacked_data = []
    for _, row in packed_df.iterrows():
        item_id = row['item_id']
        starting_date = pd.to_datetime(row['timestamp'])
        values = {'item_id': [item_id] * 90, 'timestamp': pd.date_range(start=starting_date, periods=90)}
        for col in ['NET_UNIT_PRICE', 'UNIT_PRICE', 'WEEKEND', 'HOLIDAY', 'HOLIDAY_lag_1']:
            values[col] = row[col]
        df = pd.DataFrame(values)
        unpacked_data.append(df)
    unpacked_df = pd.concat(unpacked_data, ignore_index=True)
    return unpacked_df

def unpack_jsonl_data_io(packed_json_io):
    unpacked_data = []
    for line in packed_json_io.getvalue().split('\n'):  # Get value from StringIO and split into lines
        if line.strip():  # Skip empty lines
            try:
                item = json.loads(line)
                item_id = item['item_id']
                starting_date = pd.to_datetime(item['timestamp'])
                data = {
                    'item_id': [item_id] * len(item['NET_UNIT_PRICE']),
                    'timestamp': pd.date_range(start=starting_date, periods=len(item['NET_UNIT_PRICE'])),
                    'NET_UNIT_PRICE': item['NET_UNIT_PRICE'],
                    'UNIT_PRICE': item['UNIT_PRICE'],
                    'WEEKEND': item['WEEKEND'],
                    'HOLIDAY': item['HOLIDAY'],
                    'HOLIDAY_lag_1': item['HOLIDAY_lag_1']
                }
                df = pd.DataFrame(data)
                unpacked_data.append(df)
            except Exception as e:
                print(f"Error unpacking line: {line}. Error: {e}")
    unpacked_df = pd.concat(unpacked_data, ignore_index=True)
    return unpacked_df


def transform_fn(
    model, request_body, input_content_type, output_content_type="application/json"
):
    logger.info("Transform function called with input_content_type: %s", input_content_type)
    if input_content_type == "text/csv":
        buf = StringIO(request_body)
        try:
            data = unpack_data(pd.read_csv(buf,names=['item_id','timestamp']+ model.known_covariates_names,low_memory=False))
            #data = pd.read_csv(buf,low_memory=False)
            logger.info(f"Data reading successful. Cols are...{data.columns}")
            #logger.info(data.head(3))
        except Exception as e:
            logger.error(f"Error while reading data in to df:{e}")
            return json.loads("Error while reading data"), output_content_type
    elif input_content_type == "application/json":
        #buf = StringIO(request_body)
        #data = pd.read_json(buf)
        try:
            if isinstance(request_body, bytearray):
                request_body = request_body.decode('utf-8')
            buf = StringIO(request_body)
            data = unpack_jsonl_data_io(buf)
            logger.info(f"Data reading successful. Cols are...{data.columns}")
            #logger.info(data.head(3))
        except Exception as e:
            logger.error(f"Error while reading data in to df:{e}")
            
            
    elif input_content_type == "application/jsonl":
        #buf = StringIO(request_body)
        #data = pd.read_json(buf, orient="records", lines=True)
        try:
            if isinstance(request_body, bytearray):
                request_body = request_body.decode('utf-8')
            buf = StringIO(request_body)
            #data = unpack_data(pd.read_json(buf, orient="records", lines=True))
            data = unpack_jsonl_data_io(buf)
            logger.info(f"Data reading successful. Cols are...{data.columns}")
            logger.info(data.head(3))
        except Exception as e:
            logger.error(f"Error while reading data in to df:{e}")

    else:
        raise ValueError(f"{input_content_type} input content type not supported.")
    logger.info("starting conversion to Known Covariates")
    try:
        known_covariates = TimeSeriesDataFrame.from_data_frame(data)
    except Exception as e:
        logger.error(f"Error while converting to timeseries DF: {e}")
        logger.info(data.head(2))
        
    logger.info("converted to Timeseries DF")
    grouped_sizes = known_covariates.groupby(level=0).size()
    logger.info(f"Are there any items with count < or > 90?:{(grouped_sizes < 90).any() or (grouped_sizes > 90).any()}")
    discrepant_items = grouped_sizes[(grouped_sizes < 90) | (grouped_sizes > 90)]
    discrepant_count = len(discrepant_items)
    # Filter out items with count exactly equal to 90
    filtered_items = grouped_sizes[grouped_sizes == 90].index

    # Filter the DataFrame to keep only items with count exactly equal to 90
    known_covariates = known_covariates.loc[known_covariates.index.get_level_values('item_id').isin(filtered_items)]

    if discrepant_count > 0:
        logger.info(f"Total {discrepant_count} items with count < or > 90:")
        logger.info(discrepant_items)
    else:
        logger.info("No items found with count < or > 90")
    logger.info(f"sample values: {known_covariates.head(3)}")
    
    train_data = TimeSeriesDataFrame.from_pickle(f'{model_dir}/utils/data/train.pkl')
    kvc_item_list = known_covariates.index.get_level_values('item_id').unique().tolist()
    train_data = train_data.loc[kvc_item_list]
    logger.info(f"train data loaded.. cols are {train_data.columns}")
    logger.info(f"known covariates loaded.. cols are {known_covariates.columns}")
    try:
        logger.info("Doing predictions..")
        prediction = model.predict(train_data, known_covariates=known_covariates,random_seed=123)
        logger.info('Predictions from model returned')
        #prediction = known_covariates
    except Exception as e:
        error_message = f"caught exception {e}"
        logger.error(error_message)
        # Return error message
        return json.dumps(error_message), "application/json"
    
    
    
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if "application/json" in output_content_type:
        output = prediction.reset_index().to_json()
        output_content_type = "application/json"
    elif "text/csv" in output_content_type:
        output = prediction.reset_index().to_csv(index=None)
        output_content_type = "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")
    
    logger.info("post processing done.. returning")
    return output, output_content_type
