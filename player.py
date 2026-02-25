
import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self.model_id = "marioskon/chess-smollm2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def _best_legal_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        prompt = f"FEN: {fen}\nMove:"
        best_move, best_score = None, float("-inf")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            for move in legal_moves:
                move_ids = self.tokenizer(" " + move, return_tensors="pt").input_ids.to(self.device)
                full_ids = torch.cat([inputs.input_ids, move_ids], dim=1)
                out = self.model(full_ids)
                logits = out.logits[0, inputs.input_ids.shape[1]-1:-1]
                score = torch.nn.functional.log_softmax(logits, dim=-1)
                score = score.gather(1, move_ids[0].unsqueeze(1)).sum().item()
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
            return self._best_legal_move(fen)
        except Exception:
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            return random.choice(moves).uci() if moves else None
