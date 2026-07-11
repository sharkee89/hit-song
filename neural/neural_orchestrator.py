import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TypedDict, Optional, Any, cast

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("❌ Greška: LangGraph nije instaliran. Pokreni: pip install langgraph")
    sys.exit(1)


class HitPredictionModel(nn.Module):
    def __init__(self):
        super(HitPredictionModel, self).__init__()
        self.linear = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x) / 0.8)


class OrchestratorState(TypedDict):
    file_path: str
    artist_name: str
    audio_signal: Optional[float]
    context_signal: Optional[float]
    trend_signal: Optional[float]
    viral_signal: Optional[float]
    final_hit_score: Optional[float]


class NeuralOrchestrator:
    def __init__(self):
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from neural_audio_agent import NeuralAudioAgent
        from neural_context_agent import NeuralContextAgent
        from neural_trend_agent import NeuralTrendAgent
        from neural_viral_agent import NeuralViralAgent

        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('GOOGLE_AI_API_KEY')

        self.audio_agent = NeuralAudioAgent(api_key)
        self.context_agent = NeuralContextAgent()
        self.trend_agent = NeuralTrendAgent()
        self.viral_agent = NeuralViralAgent()

        self.model = HitPredictionModel()

        with torch.no_grad():
            self.model.linear.weight = nn.Parameter(torch.tensor([[0.4, 0.5, 0.1, 0.2]]))
            self.model.linear.bias = nn.Parameter(torch.tensor([-1.2]))

        self.graph = self._compile_graph()

    def _init_node(self, state: OrchestratorState) -> Any:
        print(f"\n🚀 [Graph]: Inicijalizacija analize za numeru: {state['file_path']}")
        return {}

    def _audio_node(self, state: OrchestratorState) -> Any:
        print("🎵 [Node -> AudioAgent]: Pokrećem multimodalnu analizu...")
        report = self.audio_agent.process_track(state["file_path"])
        return {"audio_signal": report.get("activation_value", 0.5)}

    def _context_node(self, state: OrchestratorState) -> Any:
        print("📊 [Node -> ContextAgent]: Pretražujem tržišni autoritet izvođača...")
        report = self.context_agent.get_artist_report(state["artist_name"])
        return {"context_signal": report.get("activation_value", 0.5)}

    def _trend_node(self, state: OrchestratorState) -> Any:
        print("📈 [Node -> TrendAgent]: Računam geometrijsku udaljenost od trenda...")
        report = self.trend_agent.process_track(state["file_path"])
        return {"trend_signal": report.get("activation_value", 0.5)}

    def _viral_node(self, state: OrchestratorState) -> Any:
        print("🔥 [Node -> ViralAgent]: Skeniram segmente za viralni potencijal...")
        report = self.viral_agent.process_track(state["file_path"])
        return {"viral_signal": report.get("activation_value", 0.5)}

    def _pytorch_orchestrator_node(self, state: OrchestratorState) -> Any:
        print("\n🧠 [Node -> PyTorch Model]: Svi agenti su završili. Pokrećem fuziju...")
        x = torch.tensor([[
            state["audio_signal"] if state["audio_signal"] is not None else 0.5,
            state["context_signal"] if state["context_signal"] is not None else 0.5,
            state["trend_signal"] if state["trend_signal"] is not None else 0.5,
            state["viral_signal"] if state["viral_signal"] is not None else 0.5
        ]], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x)

        score = float(output.item())
        print(f"✅ [Prediction Completed]: Verovatnoća hita iznosi: {score:.4f}")
        return {"final_hit_score": score}

    def _compile_graph(self):
        """Metoda koja sklapa graf - bez komplikovanih i nestabilnih unutrašnjih retry uvoza."""
        workflow = StateGraph(OrchestratorState)  # type: ignore

        # Čisti i stabilni pozivi bez framework-specific parametara
        workflow.add_node("init", self._init_node)  # type: ignore
        workflow.add_node("audio_agent", self._audio_node)  # type: ignore
        workflow.add_node("context_agent", self._context_node)  # type: ignore
        workflow.add_node("trend_agent", self._trend_node)  # type: ignore
        workflow.add_node("viral_agent", self._viral_node)  # type: ignore
        workflow.add_node("pytorch_model", self._pytorch_orchestrator_node)  # type: ignore

        workflow.set_entry_point("init")

        workflow.add_edge("init", "audio_agent")
        workflow.add_edge("init", "context_agent")
        workflow.add_edge("init", "trend_agent")
        workflow.add_edge("init", "viral_agent")

        workflow.add_edge("audio_agent", "pytorch_model")
        workflow.add_edge("context_agent", "pytorch_model")
        workflow.add_edge("trend_agent", "pytorch_model")
        workflow.add_edge("viral_agent", "pytorch_model")

        workflow.add_edge("pytorch_model", END)

        return workflow.compile()

    def predict_song(self, file_path: str, artist_name: str) -> dict:
        initial_state = cast(OrchestratorState, {
            "file_path": file_path,
            "artist_name": artist_name,
            "audio_signal": None,
            "context_signal": None,
            "trend_signal": None,
            "viral_signal": None,
            "final_hit_score": None
        })
        return self.graph.invoke(initial_state)  # type: ignore


if __name__ == "__main__":
    try:
        orchestrator = NeuralOrchestrator()
        test_file = "/Users/admin/Downloads/National Anthem of Andorra.mp3"
        test_artist = "Taylor Swift"

        if os.path.exists(test_file):
            final_state = orchestrator.predict_song(test_file, test_artist)
            print("\n=== [KONAČAN REZULTAT IZ DELJENOG STANJA GRAFA] ===")
            print(json.dumps(final_state, indent=2))
    except Exception as e:
        print(f"Error: {e}")