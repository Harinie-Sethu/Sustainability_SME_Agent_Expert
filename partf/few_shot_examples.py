"""
Few-Shot Example Collections for Environmental SME
Organized examples for different tasks
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class FewShotLibrary:
    """
    Library of few-shot examples for prompt engineering.
    """
    
    def __init__(self, storage_dir: str = "partf/experiments/few_shots"):
        """
        Initialize few-shot library.
        
        Args:
            storage_dir: Directory to store examples
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Few-shot examples by task
        self.examples = {
            "qa": self._init_qa_examples(),
            "quiz": self._init_quiz_examples(),
            "summarization": self._init_summarization_examples(),
            "reasoning": self._init_reasoning_examples()
        }
    
    def _init_qa_examples(self) -> List[Dict[str, str]]:
        """Initialize Q&A few-shot examples."""
        return [
            {
                "input": "What is renewable energy?",
                "output": "Renewable energy is energy derived from natural sources that replenish themselves over short periods of time, such as solar radiation, wind, rain, tides, and geothermal heat. Unlike fossil fuels, which take millions of years to form, renewable energy sources are naturally and continuously renewed. The key advantage is that they produce little to no greenhouse gas emissions, making them crucial for combating climate change. Common types include solar power (from sunlight), wind power (from air movement), hydroelectric power (from flowing water), geothermal energy (from Earth's heat), and biomass energy (from organic materials)."
            },
            {
                "input": "How does composting help the environment?",
                "output": "Composting benefits the environment in several ways: (1) It reduces landfill waste by diverting organic materials that would otherwise decompose anaerobically and produce methane, a potent greenhouse gas. (2) It creates nutrient-rich soil amendment that reduces the need for chemical fertilizers, which require significant energy to produce and can pollute waterways. (3) Compost-enriched soil retains more water, reducing irrigation needs. (4) It supports soil biodiversity by providing habitat for beneficial microorganisms. (5) By completing the natural nutrient cycle, composting reduces our carbon footprint and supports sustainable agriculture."
            },
            {
                "input": "What causes ocean acidification?",
                "output": "Ocean acidification is primarily caused by the ocean's absorption of excess carbon dioxide (CO2) from the atmosphere. When CO2 dissolves in seawater, it forms carbonic acid, which lowers the ocean's pH level, making it more acidic. Since the Industrial Revolution, the ocean has absorbed about 30% of human-produced CO2 emissions. This process reduces the availability of carbonate ions, which marine organisms like corals, mollusks, and some plankton need to build their calcium carbonate shells and skeletons. The current rate of acidification is approximately 10 times faster than any period in the last 55 million years, posing severe threats to marine ecosystems and the food chains that depend on them."
            }
        ]
    
    def _init_quiz_examples(self) -> List[Dict[str, Any]]:
        """Initialize quiz generation few-shot examples."""
        return [
            {
                "input": {
                    "topic": "solar energy",
                    "difficulty": "easy",
                    "num_questions": 1
                },
                "output": {
                    "question": "What is the primary source of solar energy?",
                    "options": {
                        "A": "The Moon",
                        "B": "The Sun",
                        "C": "Wind",
                        "D": "Water"
                    },
                    "correct_answer": "B",
                    "explanation": "Solar energy comes from the Sun. Solar panels convert sunlight into electricity through photovoltaic cells. The Sun provides an abundant and renewable source of energy that can be harnessed for various applications."
                }
            },
            {
                "input": {
                    "topic": "recycling",
                    "difficulty": "medium",
                    "num_questions": 1
                },
                "output": {
                    "question": "Which material can typically be recycled the most times without losing quality?",
                    "options": {
                        "A": "Plastic",
                        "B": "Paper",
                        "C": "Glass",
                        "D": "Cardboard"
                    },
                    "correct_answer": "C",
                    "explanation": "Glass can be recycled infinitely without loss of quality or purity. Unlike paper and plastic, which degrade with each recycling cycle, glass maintains its integrity. This makes it one of the most sustainable packaging materials when properly recycled."
                }
            },
            {
                "input": {
                    "topic": "climate change",
                    "difficulty": "hard",
                    "num_questions": 1
                },
                "output": {
                    "question": "Which feedback loop in climate change involves the reduction of Earth's albedo effect?",
                    "options": {
                        "A": "Ocean acidification cycle",
                        "B": "Ice-albedo feedback",
                        "C": "Carbon-temperature feedback",
                        "D": "Methane release cycle"
                    },
                    "correct_answer": "B",
                    "explanation": "The ice-albedo feedback is a positive feedback loop where melting ice (which reflects sunlight) exposes darker ocean or land surfaces that absorb more solar radiation. This absorption causes additional warming, leading to more ice melt. Albedo refers to the reflectivity of a surface - ice has high albedo (reflects ~80% of sunlight) while ocean water has low albedo (reflects ~6%). This creates an accelerating cycle that amplifies global warming."
                }
            }
        ]
    
    def _init_summarization_examples(self) -> List[Dict[str, str]]:
        """Initialize summarization few-shot examples."""
        return [
            {
                "input": "Deforestation is the permanent removal of trees to make room for something besides forest. This can include clearing the land for agriculture or grazing, or using the timber for fuel, construction or manufacturing. Forests cover about 30% of the planet, but deforestation is clearing these essential habitats on a massive scale. The biggest driver of deforestation is agriculture. Farmers cut forests to provide more room for planting crops or grazing livestock. Often small farmers will clear a few acres by cutting down trees and burning them in a process known as slash and burn agriculture. Logging operations provide the world's wood and paper products. While some companies practice sustainable forestry, others are less scrupulous. Clear-cutting is when large areas are cut down all at once.",
                "output": "Deforestation, the permanent removal of forests, is driven primarily by agriculture, with farmers clearing land for crops and livestock. Logging for wood and paper products is another major cause. While forests cover 30% of Earth, these essential habitats are being cleared at a massive scale through practices like slash-and-burn agriculture and clear-cutting."
            },
            {
                "input": "The greenhouse effect is a natural process that warms the Earth's surface. When the Sun's energy reaches the Earth's atmosphere, some of it is reflected back to space and the rest is absorbed and re-radiated by greenhouse gases. The absorbed energy warms the atmosphere and the surface of the Earth. This process maintains the Earth's temperature at around 33 degrees Celsius warmer than it would otherwise be, allowing life on Earth to exist. However, human activities have increased the concentration of greenhouse gases, enhancing the greenhouse effect and causing global warming.",
                "output": "The greenhouse effect is a natural process where greenhouse gases trap heat in Earth's atmosphere, maintaining temperatures suitable for life - about 33°C warmer than without this effect. However, human activities have increased greenhouse gas concentrations, intensifying this effect and causing global warming."
            }
        ]
    
    def _init_reasoning_examples(self) -> List[Dict[str, Any]]:
        """Initialize reasoning few-shot examples."""
        return [
            {
                "input": "Why should we reduce plastic usage?",
                "output": {
                    "understanding": "The question asks for reasons to minimize plastic consumption and use.",
                    "key_factors": [
                        "Environmental pollution",
                        "Wildlife impact",
                        "Non-biodegradable nature",
                        "Microplastic contamination",
                        "Resource consumption in production"
                    ],
                    "reasoning": [
                        "Plastic takes 450-1000 years to decompose, accumulating in environments",
                        "8 million tons of plastic enter oceans yearly, harming marine life",
                        "Microplastics enter food chains, affecting human health",
                        "Plastic production consumes fossil fuels and releases emissions",
                        "Recycling rates are low (only ~9% globally), most ends in landfills"
                    ],
                    "conclusion": "Reducing plastic usage is crucial because plastic pollution harms ecosystems, wildlife, and human health while contributing to climate change through its production and persistence in the environment."
                }
            },
            {
                "input": "Compare solar panels vs wind turbines for home energy",
                "output": {
                    "comparison": {
                        "solar_panels": {
                            "pros": ["Works in more locations", "Lower maintenance", "Quieter", "Roof installation possible"],
                            "cons": ["Weather dependent", "Night downtime", "Higher initial cost per kW"]
                        },
                        "wind_turbines": {
                            "pros": ["24/7 generation possible", "More energy per unit in windy areas", "Lower cost per kW"],
                            "cons": ["Requires consistent wind", "Noise concerns", "Zoning restrictions", "More maintenance"]
                        }
                    },
                    "recommendation": "For most homes, solar panels are more practical due to easier installation, fewer restrictions, and broader applicability. Wind turbines suit rural properties with consistent strong winds and adequate space."
                }
            }
        ]
    
    def get_examples(self, task: str, num_examples: int = 3) -> List[Dict]:
        """
        Get few-shot examples for a task.
        
        Args:
            task: Task type
            num_examples: Number of examples to return
            
        Returns:
            List of examples
        """
        if task not in self.examples:
            return []
        return self.examples[task][:num_examples]
    
    def format_few_shot_prompt(self, task: str, num_examples: int, new_input: Any) -> str:
        """
        Format a few-shot prompt with examples.
        
        Args:
            task: Task type
            num_examples: Number of examples to include
            new_input: New input to process
            
        Returns:
            Formatted prompt with examples
        """
        examples = self.get_examples(task, num_examples)
        
        prompt_parts = ["Here are some examples:\n"]
        
        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {ex['input']}")
            prompt_parts.append(f"Output: {json.dumps(ex['output'], indent=2) if isinstance(ex['output'], dict) else ex['output']}")
            prompt_parts.append("")
        
        prompt_parts.append(f"\nNow process this input:")
        prompt_parts.append(f"Input: {new_input}")
        prompt_parts.append(f"Output:")
        
        return "\n".join(prompt_parts)
    
    def save_examples(self):
        """Save examples to disk."""
        for task, examples in self.examples.items():
            filepath = self.storage_dir / f"{task}_examples.json"
            with open(filepath, 'w') as f:
                json.dump(examples, f, indent=2)
    
    def add_example(self, task: str, input_data: Any, output_data: Any):
        """Add a new example to the library."""
        if task not in self.examples:
            self.examples[task] = []
        
        self.examples[task].append({
            "input": input_data,
            "output": output_data
        })
