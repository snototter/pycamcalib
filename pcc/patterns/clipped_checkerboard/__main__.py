from . import *
if __name__ == '__main__':
    board = ClippedCheckerboardSpecification('ccb-5x9', num_squares_horizontal=5,
                                  num_squares_vertical=9,
                                  checkerboard_square_length_mm=50)
    from vito import imvis

    imvis.imshow(board.image(), title='Calibration Board', wait_ms=-1)

    detector = ClippedCheckerboardDetector(board)
    image_points, object_points = detector.process(board.image())
    print('img points', image_points.shape)
    print('ref points', object_points.shape)
    print('img', image_points[:4,0,:])
    print('obj', object_points[0, :4, :])

    board = ClippedCheckerboardSpecification('ccb-6x8', num_squares_horizontal=6,
                                  num_squares_vertical=8,
                                  checkerboard_square_length_mm=30)
    from .. import export_board
    export_board(board, prevent_overwrite=False)
