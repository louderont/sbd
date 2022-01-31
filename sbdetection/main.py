import click
import pandas as pd

from .utils import FileReader
from .preprocessing.utils import word_tokenize
from .preprocessing.preprocessor import feature_extraction
from .models import call_model


@click.command()
@click.option('--inputpath', default='test_text.txt', help='text file to read from')
@click.option('--outputpath', default='sentences.txt', help='text file to write results to')
@click.option('--modelname', default='ML', type=click.Choice(['baseline', 'ML']), help='model to use to predict')
@click.option('--modelpath', default='model_rf.pkl', help='path of the model to use to predict')
def run(inputpath:
    str, outputpath:str, modelname:str, modelpath:str) -> None:
    """Entry point."""
    print(f'Running {modelname} model over {inputpath} data')

    # read data
    reader = FileReader(input=inputpath, output=outputpath)
    data = reader.read()

    # generate sentence boundary
    model = call_model(kind=modelname, path=modelpath)
    if modelname == 'baseline':
        sentences = model.predict(data)
    else:
        shifts = {"current": 0, "next": -1}
        tokens_to_predict = pd.DataFrame({'token':word_tokenize(' '.join(data))})
        df = feature_extraction(tokens_to_predict, shifts)
        sentences = model.predict(df)

    # output data
    print(f'Writing the {len(sentences)} extracted sentences into {outputpath}')
    reader.write(sentences)
