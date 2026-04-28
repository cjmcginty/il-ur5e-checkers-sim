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

        self.create_subscription(
            String,
            "/checkers/piece_states",
            self.piece_states_callback,
            10,
        )

        self.legal_moves = []
        self.selected_square = None
        self.hover_square = None
        self.pieces = {}

        self.root = None
        self.status = None
        self.buttons = {}
        self.ui_initialized = False
        self.gui_timer = None

    def init_ui(self):
        self.root = tk.Tk()
        self.root.title("Checkers Move Selector")

        self.status = tk.Label(self.root, text="Click a piece square")
        self.status.grid(row=0, column=0, columnspan=8)

        self.buttons = {}

        for row in range(8):
            for col in range(8):
                playable = (row + col) % 2 == 1

                btn = tk.Button(
                    self.root,
                    text="",
                    width=7,
                    height=3,
                    command=(lambda r=row, c=col: self.square_clicked(r, c)) if playable else None,
                    state="normal" if playable else "disabled",
                    disabledforeground="white",
                    activeforeground="black",
                )
                btn.bind("<Enter>", lambda _event, r=row, c=col: self.square_hovered(r, c))
                btn.bind("<Leave>", lambda _event: self.square_unhovered())
                btn.grid(row=row + 1, column=col)
                self.buttons[(row, col)] = btn

        self.clear_btn = tk.Button(
            self.root,
            text="Clear",
            command=self.clear_selection,
        )
        self.clear_btn.grid(row=9, column=0, columnspan=8, sticky="ew")

        self.reset_colors()
        self.update_piece_text()

        self.gui_timer = self.create_timer(0.05, self.update_gui)
        self.ui_initialized = True

    def legal_moves_callback(self, msg):
        try:
            self.legal_moves = json.loads(msg.data)
        except json.JSONDecodeError:
            self.legal_moves = []

        if not self.ui_initialized:
            self.init_ui()

        self.status.config(text=f"Legal moves: {len(self.legal_moves)}")

        if self.selected_square is None:
            self.reset_colors()
        else:
            self.highlight_destinations(self.selected_square)

    def piece_states_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        new_pieces = {}

        board_center_x = 0.6
        board_center_y = 0.0
        board_size = 0.40
        square_size = board_size / 8.0

        for piece in data:
            try:
                name = piece["name"]
                x = float(piece["position"]["x"])
                y = float(piece["position"]["y"])
            except (KeyError, TypeError, ValueError):
                continue

            col = int((x - (board_center_x - board_size / 2)) / square_size)
            row = int(((board_center_y + board_size / 2) - y) / square_size)

            if not (0 <= row < 8 and 0 <= col < 8):
                continue

            square = f"{row},{col}"

            text = "K" if "king" in name else "O"

            if "red" in name:
                color = "#731a1a"
            elif "black" in name:
                color = "#1f1f1f"
            else:
                color = None

            new_pieces[square] = {
                "text": text,
                "color": color,
            }

        self.pieces = new_pieces

        if self.ui_initialized:
            if self.selected_square is None:
                self.update_piece_text()
                self.reset_colors()
            else:
                self.update_piece_text()
                self.highlight_destinations(self.selected_square)

    def update_piece_text(self):
        for (row, col), btn in self.buttons.items():
            square = f"{row},{col}"
            piece = self.pieces.get(square)

            if piece is None:
                btn.config(text="")
            else:
                btn.config(
                    text=piece["text"].upper(),
                    bg=piece["color"],
                    fg="white",
                    disabledforeground="white",
                )

    def square_clicked(self, row, col):
        square = f"{row},{col}"

        if self.selected_square is None:
            starts = [m.split(" -> ")[0] for m in self.legal_moves]

            if square not in self.pieces or square not in starts:
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
        self.reset_colors(apply_hover=False)

        for move in self.legal_moves:
            if move.startswith(start_square + " -> "):
                dst = move.split(" -> ")[1]
                r, c = map(int, dst.split(","))
                self.buttons[(r, c)].config(
                    bg="lightgreen",
                    fg="black",
                    disabledforeground="black",
                )

        sr, sc = map(int, start_square.split(","))
        self.buttons[(sr, sc)].config(
            bg="yellow",
            fg="black",
            disabledforeground="black",
        )

        self.update_button_states()
        self.apply_hover_highlight()

    def reset_colors(self, apply_hover=True):
        for (row, col), btn in self.buttons.items():
            square = f"{row},{col}"
            piece = self.pieces.get(square)

            if piece is not None:
                btn.config(
                    bg=piece["color"],
                    fg="white",
                    disabledforeground="white",
                )
            elif (row + col) % 2 == 0:
                btn.config(
                    bg="white",
                    fg="black",
                    disabledforeground="black",
                )
            else:
                btn.config(
                    bg="gray",
                    fg="black",
                    disabledforeground="black",
                )

        self.update_button_states()

        if apply_hover:
            self.apply_hover_highlight()

    def update_button_states(self):
        movable_starts = set(m.split(" -> ")[0] for m in self.legal_moves)

        legal_destinations = set()
        if self.selected_square is not None:
            for move in self.legal_moves:
                if move.startswith(self.selected_square + " -> "):
                    legal_destinations.add(move.split(" -> ")[1])

        for (row, col), btn in self.buttons.items():
            square = f"{row},{col}"
            playable = (row + col) % 2 == 1

            if not playable:
                btn.config(state="disabled")
            elif self.selected_square is None:
                if square in self.pieces and square in movable_starts:
                    btn.config(state="normal")
                else:
                    btn.config(state="disabled")
            else:
                if square == self.selected_square or square in legal_destinations:
                    btn.config(state="normal")
                else:
                    btn.config(state="disabled")

    def square_hovered(self, row, col):
        self.hover_square = f"{row},{col}"
        self.apply_hover_highlight()

    def square_unhovered(self):
        self.hover_square = None

        if self.selected_square is None:
            self.reset_colors()
        else:
            self.highlight_destinations(self.selected_square)

    def apply_hover_highlight(self):
        if self.hover_square is None:
            return

        try:
            row, col = map(int, self.hover_square.split(","))
        except ValueError:
            return

        btn = self.buttons.get((row, col))
        if btn is None or btn.cget("state") == "disabled":
            return

        btn.config(
            bg="#87cefa",
            fg="black",
            disabledforeground="black",
        )

    def clear_selection(self):
        self.selected_square = None
        self.reset_colors()
        self.status.config(text="Selection cleared. Click a piece square")

    def update_gui(self):
        if self.root is None:
            return

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
        if node.root is not None:
            node.root.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
