import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets


def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()

    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='Muse-023B')

    args = parser.parse_args()

    params = BrainFlowInputParams()

    params.serial_number = args.serial_number

    # board = BoardShim(args.board_id, params)
    board = BoardShim(BoardIds.MUSE_2_BOARD, params)
    board.prepare_session()
    board.config_board("p50")
    board.add_streamer("file://default_from_streamer.csv:w",  BrainFlowPresets.DEFAULT_PRESET)
    board.add_streamer("file://aux_from_streamer.csv:w", BrainFlowPresets.AUXILIARY_PRESET)
    board.add_streamer("file://anc_from_streamer.csv:w", BrainFlowPresets.ANCILLARY_PRESET)
    board.start_stream ()
    time.sleep(10)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data_default = board.get_board_data(BrainFlowPresets.DEFAULT_PRESET)  # get all data and remove it from internal buffer
    data_aux = board.get_board_data(BrainFlowPresets.AUXILIARY_PRESET) 
    data_anc = board.get_board_data(BrainFlowPresets.ANCILLARY_PRESET) 
    board.stop_stream()
    board.release_session()

    # board.write_file(data_default, "default.csv", "w")
    # board.write_file(data_aux, "aux.csv", "w")
    # board.write_file(data_anc, "anc.csv", "w")

    #print(data)


if __name__ == "__main__":
    main() 
    