from file_preprocessing import preprocess_bad_file, save_preprocessed_file
from data_preprocessing import preprocess_data
from data_reader import read_data


def main():
    # data = preprocess_bad_file('result.tsv')
    # save_preprocessed_file(data, 'result.cropped.tsv')
    data = read_data('result.cropped.tsv')
    data = preprocess_data(data)


if __name__ == '__main__':
    main()
