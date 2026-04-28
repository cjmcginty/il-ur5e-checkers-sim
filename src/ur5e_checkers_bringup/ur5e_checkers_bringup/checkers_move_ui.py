import json
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CheckersMoveUI(Node):
    def __init__(self):
        super().__init__("checkers_move_ui")

        self.pub = self.create_publisher(
            String,
            "/checkers/selected_player_move",
            10,
        )

        self.create_subscription(
            String,
            "/checkers/legal_moves",
            self.legal_moves_callback,
            10,
        )

        self.legal_moves = []
        self.selected_square = None

        self.root = tk.Tk()
        self.root.title("Checkers Move Selector")

        self.status = tk.Label(self.root, text="Click a piece square")
        self.status.grid(row=0, column=0, columnspan=8)

        self.buttons = {}

        for row in range(8):
            for col in range(8):
                btn = tk.Button(
                    self.root,
                    text=f"{row},{col}",
                    width=7,
                    height=3,
                    command=lambda r=row, c=col: self.square_clicked(r, c),
                )
                btn.grid(row=row + 1, column=col)
                self.buttons[(row, col)] = btn

        self.clear_btn = tk.Button(
            self.root,
            text="Clear",
            command=self.clear_selection,
        )
        self.clear_btn.grid(row=9, column=0, columnspan=8, sticky="ew")

        self.timer = self.create_timer(0.05, self.update_gui)

    def legal_moves_callback(self, msg):
        try:
            self.legal_moves = json.loads(msg.data)
            self.status.config(text=f"Legal moves: {len(self.legal_moves)}")
        except json.JSONDecodeError:
            self.legal_moves = []

    def square_clicked(self, row, col):
        square = f"{row},{col}"

        if self.selected_square is None:
            starts = [m.split(" -> ")[0] for m in self.legal_moves]

            if square not in starts:
                self.status.config(text=f"No legal move starts at {square}")
                return

            self.selected_square = square
            self.status.config(text=f"Selected {square}. Now click destination.")
            self.highlight_destinations(square)
            return

        move = f"{self.selected_square} -> {square}"

        if move not in self.legal_moves:
            self.status.config(text=f"Illegal move: {move}")
            return

        msg = String()
        msg.data = move
        self.pub.publish(msg)

        self.status.config(text=f"Published: {move}")
        self.clear_selection()

    def highlight_destinations(self, start_square):
        self.reset_colors()

        for move in self.legal_moves:
            if move.startswith(start_square + " -> "):
                dst = move.split(" -> ")[1]
                r, c = map(int, dst.split(","))
                self.buttons[(r, c)].config(bg="lightgreen")

        sr, sc = map(int, start_square.split(","))
        self.buttons[(sr, sc)].config(bg="yellow")

    def reset_colors(self):
        for (row, col), btn in self.buttons.items():
            if (row + col) % 2 == 0:
                btn.config(bg="white")
            else:
                btn.config(bg="gray")

    def clear_selection(self):
        self.selected_square = None
        self.reset_colors()

    def update_gui(self):
        self.root.update_idletasks()
        self.root.update()


def main(args=None):
    rclpy.init(args=args)
    node = CheckersMoveUI()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.root.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()