from . import DAIBoardSpecification, DAIDetector

if __name__ == '__main__':
    board = DAIBoardSpecification('dai-5x9', board_width_mm=160, board_height_mm=200,
                                  margin_horizontal_mm=20, margin_vertical_mm=20,
                                  checkerboard_square_length_mm=40)
    from vito import imvis

    imvis.imshow(board.image(), title='Calibration Board', wait_ms=-1)

    detector = DAIDetector(board)
    image_points, object_points = detector.process(board.image())
    print('img points', image_points.shape)
    print('ref points', object_points.shape)
    print('img', image_points[:4,0,:])
    print('obj', object_points[0, :4, :])

    # from .. import export_board
    # export_board(board, 'dai-test')
