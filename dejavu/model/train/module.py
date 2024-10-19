import torch
from torch import nn
from torch.nn import functional as F
from ...utils.train import get_monotonic_threshold, init_threshold
from .similarity import CosineSimilarity, HeadwiseCosineSimilarity, L2Similarity, SADSimilarity, LocalL2Similarity, LocalCosineSimilarity
from .restoration import PassthroughRestoration, DiffRestoration, MLPRestoration
from .importance import CLSImportance, TldrImportance, ZeroTPruenImportance, NoneImportance
from .decision import ReuseThreshold, ReuseMLP, HeadwiseThreshold, TopKDecision
from .gating import SteepSigmoid, GumbelSoftmaxGating, HardGating, AdafuseGating


class ReuseModule(nn.Module):
    def __init__(
            self,
            num_heads,
            kept_num,
            decision_type='threshold', # 'threshold' or 'mlp'
            decision_mlp_inner_dim=64,
            decision_mlp_add_residual=False,
            decision_mlp_layer_pattern=None,
            decision_mlp_out_dim=1,
            decision_mlp_use_norm=False,
            decision_mlp_dropout=0.25,
            decision_mlp_share=None,
            decision_initialize=None,
            decision_hyperparam=None,
            decision_reference_type=False,
            gating_type='steep_sigmoid', # 'gumbel' or 'steep_sigmoid'
            gating_hyperparam=0.25,
            gating_scheduling=False,
            similarity_type='cosine^1',
            importance_type='cls^2',
            restoration_type='passthrough', # 'passthrough', 'diff', 'mlp'
            restoration_mlp_inner_dim=64,
            restoration_mlp_disable_bias=True,
            restoration_input_dim=768,
            threshold_act='',
            threshold_disable_monotonicity=False,
            use_compressed_info=False,
            num_codecnet_outputs=1,
            use_local_only=False,
            disable_final_tanh=False,
            disable_mask=False,
        ):
        super().__init__()
        self.kept_num = kept_num
        self.decision_type = decision_type

        ##############
        # Similarity #
        ##############
        if decision_mlp_use_norm:
            assert 'cosine' in similarity_type, "decision_mlp_use_norm only supports cosine similarity"
        if 'local_cosine' in similarity_type:
            exp = int(similarity_type.split('^')[1])
            self.similarity_module = LocalCosineSimilarity(exp=exp, return_norm=decision_mlp_use_norm)
        elif 'cosine' in similarity_type: # cosine^1
            exp = int(similarity_type.split('^')[1])
            self.similarity_module = CosineSimilarity(exp=exp, return_norm=decision_mlp_use_norm)
        elif 'headwise' in similarity_type: # headwise_cosine^1
            exp = int(similarity_type.split('^')[1])
            self.similarity_module = HeadwiseCosineSimilarity(num_heads=num_heads, exp=exp)
        elif 'l2' == similarity_type:
            self.similarity_module = L2Similarity()
        elif 'sad' == similarity_type:
            self.similarity_module = SADSimilarity()
        elif 'local_l2' == similarity_type:
            self.similarity_module = LocalL2Similarity()
        else:
            raise NotImplementedError

        ##############
        # Importance #
        ##############
        if 'cls' in importance_type:
            # cls^2
            exp = int(importance_type.split('^')[1])
            self.importance_module = CLSImportance(exp=exp)
        elif 'zerotprune' in importance_type:
            # zerotprune@30
            wpr_iter = int(importance_type.split('@')[1])
            self.importance_module = ZeroTPruenImportance(wpr_iter=wpr_iter)
        elif 'none' == importance_type:
            self.importance_module = NoneImportance()
        elif 'tldr' == importance_type:
            self.importance_module = TldrImportance()
        else:
            raise NotImplementedError

        ############
        # Decision #
        ############ 
        if decision_type == 'threshold':
            assert num_codecnet_outputs == 1, "Threshold decision module only supports single output"
            self.decision_module = ReuseThreshold(
                kept_num, 
                threshold_act=threshold_act,
                disable_monotonicity=threshold_disable_monotonicity,
                use_compressed_info=use_compressed_info,
                disable_final_tanh=disable_final_tanh,
                decision_hyperparam=decision_hyperparam,
            )
        elif decision_type == 'mlp':
            if decision_mlp_share is not None:
                self.decision_module = decision_mlp_share
            else:
                self.decision_module = ReuseMLP(
                    inner_dim=decision_mlp_inner_dim,
                    add_residual=decision_mlp_add_residual,
                    layer_pattern=decision_mlp_layer_pattern,
                    use_compressed_info=use_compressed_info,
                    decision_mlp_out_dim=decision_mlp_out_dim,
                    decision_mlp_use_norm=decision_mlp_use_norm,
                    decision_initialize=decision_initialize,
                    decision_reference_type=decision_reference_type,
                    dropout=decision_mlp_dropout,
                )
        elif decision_type == 'headwise-threshold':
            self.decision_module = HeadwiseThreshold(
                kept_num,
                num_heads,
                disable_monotonicity=threshold_disable_monotonicity,
                threshold_act=threshold_act,
                use_compressed_info=use_compressed_info,
            )
        elif decision_type == 'topk':
            self.decision_module = TopKDecision(
                decision_hyperparam,
            )
        else:
            raise NotImplementedError

        ##########
        # Gating #
        ##########
        if gating_type == 'gumbel':
            self.gating_module = GumbelSoftmaxGating(tau=gating_hyperparam)
        elif gating_type == 'steep_sigmoid':
            self.gating_module = SteepSigmoid(s=gating_hyperparam) # 40?
        elif gating_type == 'hard':
            self.gating_module = HardGating()
        elif gating_type == 'adafuse':
            self.gating_module = AdafuseGating(tau=gating_hyperparam, gating_scheduling=gating_scheduling)
        else:
            raise NotImplementedError

        ###############
        # Restoration #
        ###############
        if 'passthrough' == restoration_type:
            self.restoration_module = PassthroughRestoration()
        elif 'diff' == restoration_type:
            self.restoration_module = DiffRestoration()
        elif 'mlp' == restoration_type:
            self.restoration_module = MLPRestoration(
                input_dim=restoration_input_dim,
                inner_dim=restoration_mlp_inner_dim,
                disable_bias=restoration_mlp_disable_bias,
            )
        else:
            raise NotImplementedError

        self.use_compressed_info = use_compressed_info
        self.use_local_only = use_local_only
        self.disable_mask = disable_mask

    def forward(
            self,
            cached_states,
            pre_proj,
            hidden_states,
            query_states,
            key_states,
            value_states,
            attn_weights,
            compressed_map=None,
            ref_mask=None,
            ref_type=None,
            skip_cls=True,
            **kwargs
        ):
        B, N, dim = hidden_states.shape

        (
            cached_pre_proj,
            cached_hidden_states,
            cached_query_states,
            cached_key_states,
            cached_value_states,
        ) = cached_states

        if skip_cls:
            cls_pre_proj = pre_proj[:, 0:1]
            cls_hidden_states = hidden_states[:, 0:1]
            cls_query_states = query_states[:, 0:1]
            cls_key_states = key_states[:, 0:1]
            cls_value_states = value_states[:, 0:1]

            pre_proj = pre_proj[:, 1:]
            hidden_states = hidden_states[:, 1:]
            query_states = query_states[:, 1:]
            key_states = key_states[:, 1:]
            value_states = value_states[:, 1:]

        similarity, norm = self.similarity_module(pre_proj, cached_pre_proj)
        if not self.disable_mask:
            assert ref_mask is not None, "Mask must be provided for this model"
            similarity = similarity.view(B, N-1, -1, N)
            ref_mask = ref_mask[:, :similarity.shape[2]]
            ref_mask = ref_mask.view(B, 1, -1, 1).expand(-1, N-1, -1, N)
            similarity[~ref_mask] = -1e9
            similarity = similarity.view(B, N-1, -1)

        importance = self.importance_module(attn_weights)
        reuse_decision, most_similar_idx, _ = self.decision_module(
            importance=importance,
            similarity=similarity if norm is None else (similarity, norm),
            compressed_map=compressed_map,
            ref_type=ref_type,
        )

        # Gather most similar inputs
        most_similar_pre_proj = torch.gather(
            cached_pre_proj,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_hidden_states = torch.gather(
            cached_hidden_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_query_states = torch.gather(
            cached_query_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_key_states = torch.gather(
            cached_key_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_value_states = torch.gather(
            cached_value_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        (
            restored_hidden_states,
            restored_query_states,
            restored_key_states,
            restored_value_states,
        ) = self.restoration_module(
            current_pre_proj=pre_proj,
            most_similar_pre_proj=most_similar_pre_proj,
            most_similar_hidden_states=most_similar_hidden_states,
            most_similar_query_states=most_similar_query_states,
            most_similar_key_states=most_similar_key_states,
            most_similar_value_states=most_similar_value_states,
        )

        hard = kwargs.get('hard', False)
        tau = kwargs.get('tau', None)
        
        reuse_map, (
            pre_proj,
            hidden_states,
            query_states,
            key_states,
            value_states,
        ) = self.gating_module(
            reuse_decision,
            upper_values=(
                most_similar_pre_proj, 
                restored_hidden_states,
                restored_query_states,
                restored_key_states,
                restored_value_states,
            ),
            lower_values=(
                pre_proj,
                hidden_states,
                query_states,
                key_states,
                value_states,
            ),
            hard=hard,
            tau=tau
        )
        
        reuse_map = reuse_map.squeeze(-1)
        if skip_cls:
            # Re-attach cls token
            pre_proj = torch.cat((cls_pre_proj, pre_proj), dim=1)
            hidden_states = torch.cat((cls_hidden_states, hidden_states), dim=1)
            query_states = torch.cat((cls_query_states, query_states), dim=1)
            key_states = torch.cat((cls_key_states, key_states), dim=1)
            value_states = torch.cat((cls_value_states, value_states), dim=1)

            reuse_map = torch.cat(
                    (
                        torch.zeros((B, 1), dtype=torch.bool, device=reuse_map.device),
                        reuse_map
                    ),
                    dim=1
                )

        return reuse_map, pre_proj, hidden_states, query_states, key_states, value_states

    def forward_v2(
            self,
            cached_states,
            ref_state,
            *output_states,
            attn_weights=None,
            compressed_map=None,
            ref_mask=None,
            skip_cls=True,
            **kwargs
        ):
        B, N, dim = ref_state.shape


        if skip_cls:
            cls_ref_state = ref_state[:, 0:1]
            cls_output_states = [output_state[:, 0:1] for output_state in output_states]

            ref_state = ref_state[:, 1:]
            output_states = [output_state[:, 1:] for output_state in output_states]

        cached_ref_state, *cached_output_states = cached_states

        assert len(cached_output_states) == len(output_states), "Cached and current output states must have the same length"

        similarity = self.similarity_module(ref_state, cached_ref_state)

        if ref_mask is not None:
            similarity = similarity.view(B, N-1, -1, N)
            ref_mask = ref_mask.view(B, 1, -1, 1).expand(-1, N-1, -1, N)
            similarity[~ref_mask] = -2.
            similarity = similarity.view(B, N-1, -1)

        importance = self.importance_module(attn_weights)
        reuse_decision, most_similar_idx, _ = self.decision_module(
            importance=importance,
            similarity=similarity,
            compressed_map=compressed_map,
        )

        # Gather most similar inputs
        most_similar_ref_state = torch.gather(
            cached_ref_state,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_output_states = [
            torch.gather(
                cached_output_state,
                dim=1,
                index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
            ).squeeze(1)
            for cached_output_state in cached_output_states
        ]

        assert isinstance(self.restoration_module, PassthroughRestoration), "Only PassthroughRestoration is supported for forward_v2"

        reuse_map, (
            ref_state,
            *output_states,
        ) = self.gating_module(
            reuse_decision,
            upper_values=(
                most_similar_ref_state,
                *most_similar_output_states,
            ),
            lower_values=(
                ref_state,
                *output_states,
            ),
            hard=kwargs.get('hard', False),
            tau=kwargs.get('tau', None)
        )
        
        reuse_map = reuse_map.squeeze(-1)

        if skip_cls:
            # Re-attach cls token
            ref_state = torch.cat((cls_ref_state, ref_state), dim=1)
            output_states = [
                torch.cat((cls_output_state, output_state), dim=1)
                for cls_output_state, output_state in zip(cls_output_states, output_states)
            ]

            reuse_map = torch.cat(
                (
                    torch.zeros((B, 1), dtype=torch.bool, device=reuse_map.device),
                    reuse_map
                ),
                dim=1
            )

        return reuse_map, ref_state, *output_states