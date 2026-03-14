from ur5e_checkers_bringup.board import CheckersBoard


def main() -> None:
    board = CheckersBoard()
    legal = board.legal_moves()

    print(f"Current turn: {board.turn}")
    print(f"Legal moves: {len(legal)}")

    if legal:
        print(f"First move: {legal[0]}")
        board.apply_move(legal[0])
        print("Applied first legal move.")


if __name__ == "__main__":
    main()