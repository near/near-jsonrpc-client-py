"""ClientConfig where some fields can be updated at runtime."""

from near_jsonrpc_models.chunk_distribution_network_config import ChunkDistributionNetworkConfig
from near_jsonrpc_models.cloud_archival_writer_config import CloudArchivalWriterConfig
from near_jsonrpc_models.epoch_sync_config import EpochSyncConfig
from near_jsonrpc_models.gcconfig import GCConfig
from near_jsonrpc_models.log_summary_style import LogSummaryStyle
from near_jsonrpc_models.mutable_config_value import MutableConfigValue
from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.protocol_version_check_config import ProtocolVersionCheckConfig
from near_jsonrpc_models.state_sync_config import StateSyncConfig
from near_jsonrpc_models.tracked_shards_config import TrackedShardsConfig
from near_jsonrpc_models.version import Version
from pydantic import BaseModel
from pydantic import conint
from pydantic import conlist
from typing import List


class RpcClientConfigResponse(BaseModel):
    # Not clear old data, set `true` for archive nodes.
    archive: bool = None
    # Horizon at which instead of fetching block, fetch full state.
    block_fetch_horizon: conint(ge=0, le=18446744073709551615) = None
    # Behind this horizon header fetch kicks in.
    block_header_fetch_horizon: conint(ge=0, le=18446744073709551615) = None
    # Duration to check for producing / skipping block.
    block_production_tracking_delay: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Time between check to perform catchup.
    catchup_step_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Chain id for status.
    chain_id: str = None
    # Optional config for the Chunk Distribution Network feature.
    # If set to `None` then this node does not participate in the Chunk Distribution Network.
    # Nodes not participating will still function fine, but possibly with higher
    # latency due to the need of requesting chunks over the peer-to-peer network.
    chunk_distribution_network: ChunkDistributionNetworkConfig | None = None
    # Time between checking to re-request chunks.
    chunk_request_retry_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Number of threads for ChunkValidationActor pool.
    chunk_validation_threads: conint(ge=0, le=4294967295) = None
    # Multiplier for the wait time for all chunks to be received.
    chunk_wait_mult: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Height horizon for the chunk cache. A chunk is removed from the cache
    # if its height + chunks_cache_height_horizon < largest_seen_height.
    # The default value is DEFAULT_CHUNKS_CACHE_HEIGHT_HORIZON.
    chunks_cache_height_horizon: conint(ge=0, le=18446744073709551615) = None
    # Number of threads to execute background migration work in client.
    client_background_migration_threads: conint(ge=0, le=4294967295) = None
    # Configuration for a cloud-based archival writer. If this config is present, the writer is enabled and
    # writes chunk-related data based on the tracked shards.
    cloud_archival_writer: CloudArchivalWriterConfig | None = None
    # If true, the node won't forward transactions to next the chunk producers.
    disable_tx_routing: bool = None
    # Time between running doomslug timer.
    doomslug_step_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # If true, transactions for the next chunk will be prepared early, right after the previous chunk's
    # post-state is ready. This can help produce chunks faster, for high-throughput chains.
    # The current implementation increases latency on low-load chains, which will be fixed in the future.
    # The default is disabled.
    enable_early_prepare_transactions: bool = None
    enable_multiline_logging: bool = None
    # Re-export storage layer statistics as prometheus metrics.
    enable_statistics_export: bool = None
    # Epoch length.
    epoch_length: conint(ge=0, le=18446744073709551615) = None
    # Options for epoch sync.
    epoch_sync: EpochSyncConfig = None
    # Graceful shutdown at expected block height.
    expected_shutdown: MutableConfigValue = None
    # Garbage collection configuration.
    gc: GCConfig = None
    # Expected increase of header head height per second during header sync
    header_sync_expected_height_per_second: conint(ge=0, le=18446744073709551615) = None
    # How much time to wait after initial header sync
    header_sync_initial_timeout: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # How much time to wait after some progress is made in header sync
    header_sync_progress_timeout: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # How much time to wait before banning a peer in header sync if sync is too slow
    header_sync_stall_ban_timeout: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Period between logging summary information.
    log_summary_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Enable coloring of the logs
    log_summary_style: LogSummaryStyle = None
    # Maximum wait for approvals before producing block.
    max_block_production_delay: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Maximum duration before skipping given height.
    max_block_wait_delay: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Max burnt gas per view method.  If present, overrides value stored in
    # genesis file.  The value only affects the RPCs without influencing the
    # protocol thus changing it per-node doesn’t affect the blockchain.
    max_gas_burnt_view: NearGas | None = None
    # Minimum duration before producing block.
    min_block_production_delay: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Minimum number of peers to start syncing.
    min_num_peers: conint(ge=0, le=4294967295) = None
    # Number of block producer seats
    num_block_producer_seats: conint(ge=0, le=18446744073709551615) = None
    # Maximum size of state witnesses in the OrphanStateWitnessPool.
    # 
    # We keep only orphan witnesses which are smaller than this size.
    # This limits the maximum memory usage of OrphanStateWitnessPool.
    orphan_state_witness_max_size: conint(ge=0, le=18446744073709551615) = None
    # OrphanStateWitnessPool keeps instances of ChunkStateWitness which can't be processed
    # because the previous block isn't available. The witnesses wait in the pool until the
    # required block appears. This variable controls how many witnesses can be stored in the pool.
    orphan_state_witness_pool_size: conint(ge=0, le=4294967295) = None
    # Limit the time of adding transactions to a chunk.
    # A node produces a chunk by adding transactions from the transaction pool until
    # some limit is reached. This time limit ensures that adding transactions won't take
    # longer than the specified duration, which helps to produce the chunk quickly.
    produce_chunk_add_transactions_time_limit: str = None
    # Produce empty blocks, use `false` for testing.
    produce_empty_blocks: bool = None
    # Determines whether client should exit if the protocol version is not supported
    # for the next or next next epoch.
    protocol_version_check: ProtocolVersionCheckConfig = None
    resharding_config: MutableConfigValue = None
    # Listening rpc port for status.
    rpc_addr: str | None = None
    # Save observed instances of invalid ChunkStateWitness to the database in DBCol::InvalidChunkStateWitnesses.
    # Saving invalid witnesses is useful for analysis and debugging.
    # This option can cause extra load on the database and is not recommended for production use.
    save_invalid_witnesses: bool = None
    # Save observed instances of ChunkStateWitness to the database in DBCol::LatestChunkStateWitnesses.
    # Saving the latest witnesses is useful for analysis and debugging.
    # This option can cause extra load on the database and is not recommended for production use.
    save_latest_witnesses: bool = None
    # Whether to persist state changes on disk or not.
    save_state_changes: bool = None
    # save_trie_changes should be set to true iff
    # - archive if false - non-archival nodes need trie changes to perform garbage collection
    # - archive is true, cold_store is configured and migration to split_storage is finished - node
    # working in split storage mode needs trie changes in order to do garbage collection on hot.
    save_trie_changes: bool = None
    # Whether to persist transaction outcomes to disk or not.
    save_tx_outcomes: bool = None
    # Whether to persist partial chunk parts for untracked shards or not.
    save_untracked_partial_chunks_parts: bool = None
    # Skip waiting for sync (for testing or single node testnet).
    skip_sync_wait: bool = None
    # Number of threads for StateRequestActor pool.
    state_request_server_threads: conint(ge=0, le=4294967295) = None
    # Number of seconds between state requests for view client.
    # Throttling window for state requests (headers and parts).
    state_request_throttle_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Maximum number of state requests served per throttle period
    state_requests_per_throttle_period: conint(ge=0, le=4294967295) = None
    # Options for syncing state.
    state_sync: StateSyncConfig = None
    # Whether to use the State Sync mechanism.
    # If disabled, the node will do Block Sync instead of State Sync.
    state_sync_enabled: bool = None
    # Additional waiting period after a failed request to external storage
    state_sync_external_backoff: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # How long to wait for a response from centralized state sync
    state_sync_external_timeout: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # How long to wait for a response from p2p state sync
    state_sync_p2p_timeout: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # How long to wait after a failed state sync request
    state_sync_retry_backoff: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # How often to check that we are not out of sync.
    sync_check_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # Sync height threshold: below this difference in height don't start syncing.
    sync_height_threshold: conint(ge=0, le=18446744073709551615) = None
    # Maximum number of block requests to send to peers to sync
    sync_max_block_requests: conint(ge=0, le=4294967295) = None
    # While syncing, how long to check for each step.
    sync_step_period: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    tracked_shards_config: TrackedShardsConfig = None
    # Limit of the size of per-shard transaction pool measured in bytes. If not set, the size
    # will be unbounded.
    transaction_pool_size_limit: conint(ge=0, le=18446744073709551615) | None = None
    transaction_request_handler_threads: conint(ge=0, le=4294967295) = None
    # Upper bound of the byte size of contract state that is still viewable. None is no limit
    trie_viewer_state_size_limit: conint(ge=0, le=18446744073709551615) | None = None
    # Time to persist Accounts Id in the router without removing them.
    ttl_account_id_router: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2) = None
    # If the node is not a chunk producer within that many blocks, then route
    # to upcoming chunk producers.
    tx_routing_height_horizon: conint(ge=0, le=18446744073709551615) = None
    # Version of the binary.
    version: Version = None
    # Number of threads for ViewClientActor pool.
    view_client_threads: conint(ge=0, le=4294967295) = None
