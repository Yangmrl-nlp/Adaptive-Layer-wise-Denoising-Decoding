Reviewer #1: 

1. The proposed lightweight predictor relies solely on textual prompts to predict the optimal denoising layer, which may limit its effectiveness in cases where hallucinations are strongly driven by visual ambiguity rather than linguistic structure.

2. The quantification strategy for selecting the optimal denoising layer is token-level and supervised by ground-truth answers, which may not fully reflect realistic open-ended generation scenarios where such supervision is unavailable.

3. The method introduces a non-negligible inference latency overhead, and while the paper reports this cost, there is limited discussion on how ALD² would scale to longer responses or higher-resolution visual inputs.

4. The visual token pruning ratio is fixed across models and tasks, which may not be optimal in diverse scenarios, especially for complex scenes with fine-grained visual details.(section limitation)

5. The evaluation focuses primarily on object-level hallucinations and general benchmarks, while other forms of hallucinations such as relational, temporal, or causal hallucinations are not explicitly analyzed.

6. The paper does not sufficiently position ALD² with respect to recent work on hallucination behavior in long-form responses and vision-aware internal dynamics, which would help clarify the novelty and limitations of the approach. The discussion of limitations should explicitly acknowledge and cite the following relevant works to better contextualize the proposed approach:

\* Why LVLMs Are More Prone to Hallucinations in Longer Responses
\* Visual In-Context Learning for Large Vision-Language Models
\* Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence
\* Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback
\* Visual hallucination detection in large vision-language models



Reviewer #2: Summary

This paper proposes ALD² (Adaptive Layer-wise Denoising Decoding), a decoding-time framework to mitigate hallucinations in large vision-language models (LVLMs). The authors analyze hallucinations from a layer-wise noise propagation perspective, arguing that redundant visual tokens in shallow layers introduce noise that later overrides correct token predictions. To address this, the paper combines visual token pruning at adaptively selected shallow layers with multiplicative decoding, guided by a lightweight layer predictor. Extensive experiments on CHAIR, POPE, and MME benchmarks demonstrate consistent improvements across multiple LVLM backbones.

Strength
The layer-wise noise interpretation of hallucinations is insightful and complements existing contrastive decoding and post-hoc correction methods.

The combination of shallow-layer denoising and multiplicative decoding is intuitive and technically grounded. The adaptive layer selection further strengthens the approach.

Results on CHAIR, POPE, and MME across several LVLMs are convincing and show consistent gains over strong baselines (Greedy, DoLa, ICD, VCD).

Weakness
**The decoding latency is significantly higher than greedy decoding, and the paper could better discuss when this trade-off is acceptable in real-world deployments.
As acknowledged by the authors, ALD² cannot eliminate hallucinations when models are highly confident in incorrect predictions.
I recommend including evaluations on more recent and stronger LVLMs (e.g., Qwen-VL-2.5) to better demonstrate the generality and scalability of the proposed method.
The related work on hallucination mitigation appears incomplete. I recommend expanding the discussion to include more recent studies on hallucinations in LVLMs, particularly works focusing on grounding, faithfulness, and decoding-time mitigation. For example, **

**FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models, **

**Aligning large multimodal models with factually augmented rlhf, A Unified Hallucination Mitigation Framework for Large Vision-Language Models. **

**Mmbench: Is your multi-modal model an all-around player? Fine-grained and Explanable Factuality Evaluation for Multimodal Summarization. **

**FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation, **

**Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning, **

**FGAIF: Aligning Large Vision-Language Models with Fine-grained AI Feedback. **

**From pixels to tokens: Revisiting object hallucinations in large vision-language models.**



Reviewer #3: 

1- The authors treat hallucination as a shallow-layer noise propagation problem and attempt to denoise internal representations by pruning visual tokens at a selected shallow layer, followed by multiplying the resulting "denoised" distribution with the final-layer distribution. However, the manuscript does not formalize this design choice or explain why multiplicative fusion is preferable to alternative strategies (e.g., interpolation, logit addition with temperature scaling, KL-regularized fusion). The manuscript need an ablation study comparing multiplication versus addition and interpolation.

2- Most existing visual token pruning methods primarily aim to accelerate inference, whereas ALD uses pruning to modify the output distribution in order to reduce hallucinations. However, it is unclear whether the observed performance gains stem mainly from token pruning itself or from the proposed adaptive layer-wise selection and multiplicative fusion. Additional experiments are needed to disentangle the contribution of each component.(section Ablation)

3- Although the authors report results on POPE and MME, these benchmarks are fundamentally yes/no evaluations. The manuscript does not clearly explain how model outputs are parsed into binary decisions. For example, how are responses with additional text, hedging, or ambiguous phrasing handled? Clarifying the answer extraction protocol is essential for ensuring fair and reproducible evaluation.

4- In Appendix A, the definitions of the CHAIR metrics appear inconsistent with the standard formulation. In particular, the denominators for instance-level and sentence-level metrics seem to be swapped or mislabeled. This should be carefully checked and corrected, or the authors should explicitly state that they follow the official CHAIR evaluation implementation.

5- As stated in Section 3.2, the layer predictor is trained using prompt-layer pairs derived from ground-truth answers. This implies that ALD is not a purely inference-only method, as it requires a supervised preparation stage. This distinction should be clearly acknowledged and discussed, especially when comparing ALD to training-free decoding approaches.

6- The ablation study should be strengthened by including the following analyses: Multiplicative fusion versus additive and interpolative fusion strategies; Prune-only decoding versus the full ALD² framework, to demonstrate that pruning alone does not account for the improvements; Sensitivity analysis of the layer search range (e.g., justification for restricting the search to the first three-quarters of the model depth), ideally supported by a quantitative plot.

7- Several recent hallucination mitigation methods are missing from the related work discussion, including Med-VCD, Dynamic Correction Decoding (DeCo), Self-Introspective Decoding (SID), and Language Contrastive Decoding (LCD). Incorporating these methods would provide a more comprehensive and up-to-date comparison with the current literature.



Reviewer #4: Research summary: The research reported in this manuscript addresses the critical challenge of visual hallucinations in Large Vision-Language Models (LVLMs) by introducing a novel framework titled Adaptive Layer-wise Denoising Decoding (ALD2) .The authors operate under the hypothesis that hallucinations are not solely semantic errors but are driven by noise propagation from shallow layers, where redundant visual tokens create unstable predictions. To counter this, the authors utilizes a lightweight predictor to dynamically identify an optimal shallow layer, prunes redundant visual tokens to suppress noise, and fuses the denoised distribution with the final layer's output via multiplicative decoding.
1．Mechanism of Action: The study establishes that while shallow layers often contain correct semantic signals, they are masked by high-entropy noise caused by visual redundancy. ALD2 mitigates this by pruning the bottom 10% of visual tokens (based on attention scores) in a targeted early layer and multiplying this "denoised" probability distribution with the final layer's distribution.
2．Performance Improvements: Extensive evaluations on benchmarks such as POPE, CHAIR, and MME across multiple models demonstrate that ALD2 consistently mitigates hallucinations.
Comparative Advantage: The method outperforms standard baselines, including Greedy Decoding, DoLa, Visual Contrastive Decoding , and Instruction Contrastive Decoding across tested metrics. For example, on CHAIR metrics, InstructBLIP-Vicuna-7B with ALD2 surpassed all baselines in reducing hallucinated objects.
Comparative Advantage: The method outperforms standard baselines, including Greedy Decoding, DoLa, Visual Contrastive Decoding (VCD), and Instruction Contrastive Decoding (ICD) across tested metrics.
Major Strengths: The major strengths of the research are:
1．The study shifts the focus from external input perturbations to internal model dynamics, effectively reframing hallucination as a layer-wise noise propagation problem where correct signals in shallow layers are masked by redundant visual tokens.
2．The proposed framework is rigorously tested across three representative LVLMs (LLaVA-1.5, InstructBLIP, and LLaVA-Next) and multiple diverse benchmarks (CHAIR, POPE, and MME), demonstrating consistent and significant accuracy improvements compared to strong baselines like VCD and ICD.
3．The ALD2 framework combines dynamic layer selection, visual token pruning, and multiplicative decoding to suppress noise while preserving deep-layer semantic reliability.
4．The manuscript has been improved by incorporating explicit research objectives and a dedicated section on theoretical and practical implications, directly addressing editor concerns regarding clarity and impact.
Major Weaknesses:The major weaknesses of the research are:
1．While the paper acknowledges the increased latency, the discussion regarding the trade-off between this computational cost and the performance gains could be slightly expanded to better address practical deployment scenarios.
2．The method relies on a fixed 10% ratio for visual token pruning, which is shown to be effective on average; however, a brief discussion on the potential benefits (or lack thereof) of a dynamic pruning threshold could further strengthen the "Limitations" section.
The explanation for the counter-intuitive result where the LLaVA-Next-8B model achieves high performance gains despite low predictor accuracy (around 30%) could be deepened to better illustrate the robustness of the layer selection mechanism.

Grammar and Readability: The manuscript is well-written, and the revisions have significantly improved its overall clarity and flow. The English is coherent and suitable for an academic audience. The figures and tables are clear and informative . A final round of minor proofreading is recommended to catch any lingering typographical errors.

Specific Comments: My specific comments concerning this manuscript are:
**1．The manuscript notes a 94% increase in latency compared to greedy decoding. While this is acknowledged, it would be beneficial to add a brief discussion in the "limitations" or "conclusion" section regarding potential strategies to optimize this in future work, or to more explicitly frame the trade-off for applications where accuracy is paramount over speed.
2．The divergence between the low predictor accuracy (30%) and the high performance gains on the LLaVA-Next-8B model is a compelling finding. I suggest expanding the explanation to hypothesize why this specific model benefits so significantly even when the "optimal" layer is not perfectly predicted. This would add depth to the analysis of the method's robustness.
The choice of a fixed 10% pruning ratio is supported by the ablation study. A short sentence could be added to discuss whether the sensitivity to noise varies significantly across different layers, which could inform future adaptive pruning strategies.**

Concluding Remarks:The authors have done a commendable job revising the manuscript, particularly by adding clear research objectives and a dedicated discussion on theoretical and practical implications. The proposed ALD2 framework presents a novel and effective solution for hallucination mitigation in LVLMs, supported by experimental results. I recommend that the authors address the minor comments regarding the computational latency trade-offs and the predictor's behavior on specific models to further polish the paper.