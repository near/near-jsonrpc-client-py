"""Typed view of ExtCostsConfig to preserve JSON output field names in protocol
config RPC output."""

from near_jsonrpc_models.near_gas import NearGas
from pydantic import BaseModel


class ExtCostsConfigView(BaseModel):
    # Base cost for multiexp
    alt_bn128_g1_multiexp_base: NearGas = None
    # Per element cost for multiexp
    alt_bn128_g1_multiexp_element: NearGas = None
    # Base cost for sum
    alt_bn128_g1_sum_base: NearGas = None
    # Per element cost for sum
    alt_bn128_g1_sum_element: NearGas = None
    # Base cost for pairing check
    alt_bn128_pairing_check_base: NearGas = None
    # Per element cost for pairing check
    alt_bn128_pairing_check_element: NearGas = None
    # Base cost for calling a host function.
    base: NearGas = None
    bls12381_g1_multiexp_base: NearGas = None
    bls12381_g1_multiexp_element: NearGas = None
    bls12381_g2_multiexp_base: NearGas = None
    bls12381_g2_multiexp_element: NearGas = None
    bls12381_map_fp2_to_g2_base: NearGas = None
    bls12381_map_fp2_to_g2_element: NearGas = None
    bls12381_map_fp_to_g1_base: NearGas = None
    bls12381_map_fp_to_g1_element: NearGas = None
    bls12381_p1_decompress_base: NearGas = None
    bls12381_p1_decompress_element: NearGas = None
    bls12381_p1_sum_base: NearGas = None
    bls12381_p1_sum_element: NearGas = None
    bls12381_p2_decompress_base: NearGas = None
    bls12381_p2_decompress_element: NearGas = None
    bls12381_p2_sum_base: NearGas = None
    bls12381_p2_sum_element: NearGas = None
    bls12381_pairing_base: NearGas = None
    bls12381_pairing_element: NearGas = None
    contract_compile_base: NearGas = None
    contract_compile_bytes: NearGas = None
    # Base cost of loading a pre-compiled contract
    contract_loading_base: NearGas = None
    # Cost per byte of loading a pre-compiled contract
    contract_loading_bytes: NearGas = None
    # Cost of calling ecrecover
    ecrecover_base: NearGas = None
    # Cost of getting ed25519 base
    ed25519_verify_base: NearGas = None
    # Cost of getting ed25519 per byte
    ed25519_verify_byte: NearGas = None
    # Cost of getting sha256 base
    keccak256_base: NearGas = None
    # Cost of getting sha256 per byte
    keccak256_byte: NearGas = None
    # Cost of getting sha256 base
    keccak512_base: NearGas = None
    # Cost of getting sha256 per byte
    keccak512_byte: NearGas = None
    # Cost for calling logging.
    log_base: NearGas = None
    # Cost for logging per byte
    log_byte: NearGas = None
    # Cost for calling `promise_and`
    promise_and_base: NearGas = None
    # Cost for calling `promise_and` for each promise
    promise_and_per_promise: NearGas = None
    # Cost for calling `promise_return`
    promise_return: NearGas = None
    # Cost for reading trie node from memory
    read_cached_trie_node: NearGas = None
    # Base cost for guest memory read
    read_memory_base: NearGas = None
    # Cost for guest memory read
    read_memory_byte: NearGas = None
    # Base cost for reading from register
    read_register_base: NearGas = None
    # Cost for reading byte from register
    read_register_byte: NearGas = None
    # Cost of getting ripemd160 base
    ripemd160_base: NearGas = None
    # Cost of getting ripemd160 per message block
    ripemd160_block: NearGas = None
    # Cost of getting sha256 base
    sha256_base: NearGas = None
    # Cost of getting sha256 per byte
    sha256_byte: NearGas = None
    # Storage trie check for key existence cost base
    storage_has_key_base: NearGas = None
    # Storage trie check for key existence per key byte
    storage_has_key_byte: NearGas = None
    # Create trie range iterator cost per byte of from key.
    storage_iter_create_from_byte: NearGas = None
    # Create trie prefix iterator cost base
    storage_iter_create_prefix_base: NearGas = None
    # Create trie prefix iterator cost per byte.
    storage_iter_create_prefix_byte: NearGas = None
    # Create trie range iterator cost base
    storage_iter_create_range_base: NearGas = None
    # Create trie range iterator cost per byte of to key.
    storage_iter_create_to_byte: NearGas = None
    # Trie iterator per key base cost
    storage_iter_next_base: NearGas = None
    # Trie iterator next key byte cost
    storage_iter_next_key_byte: NearGas = None
    # Trie iterator next key byte cost
    storage_iter_next_value_byte: NearGas = None
    # Storage trie read key overhead base cost, when doing large reads
    storage_large_read_overhead_base: NearGas = None
    # Storage trie read key overhead  per-byte cost, when doing large reads
    storage_large_read_overhead_byte: NearGas = None
    # Storage trie read key base cost
    storage_read_base: NearGas = None
    # Storage trie read key per byte cost
    storage_read_key_byte: NearGas = None
    # Storage trie read value cost per byte cost
    storage_read_value_byte: NearGas = None
    # Remove key from trie base cost
    storage_remove_base: NearGas = None
    # Remove key from trie per byte cost
    storage_remove_key_byte: NearGas = None
    # Remove key from trie ret value byte cost
    storage_remove_ret_value_byte: NearGas = None
    # Storage trie write key base cost
    storage_write_base: NearGas = None
    # Storage trie write cost per byte of evicted value.
    storage_write_evicted_byte: NearGas = None
    # Storage trie write key per byte cost
    storage_write_key_byte: NearGas = None
    # Storage trie write value per byte cost
    storage_write_value_byte: NearGas = None
    # Cost per reading trie node from DB
    touching_trie_node: NearGas = None
    # Base cost of decoding utf16. It's used for `log_utf16`.
    utf16_decoding_base: NearGas = None
    # Cost per byte of decoding utf16. It's used for `log_utf16`.
    utf16_decoding_byte: NearGas = None
    # Base cost of decoding utf8. It's used for `log_utf8` and `panic_utf8`.
    utf8_decoding_base: NearGas = None
    # Cost per byte of decoding utf8. It's used for `log_utf8` and `panic_utf8`.
    utf8_decoding_byte: NearGas = None
    # Cost of calling `validator_stake`.
    validator_stake_base: NearGas = None
    # Cost of calling `validator_total_stake`.
    validator_total_stake_base: NearGas = None
    # Base cost for guest memory write
    write_memory_base: NearGas = None
    # Cost for guest memory write per byte
    write_memory_byte: NearGas = None
    # Base cost for writing into register
    write_register_base: NearGas = None
    # Cost for writing byte into register
    write_register_byte: NearGas = None
    # Base cost for creating a yield promise.
    yield_create_base: NearGas = None
    # Per byte cost of arguments and method name.
    yield_create_byte: NearGas = None
    # Base cost for resuming a yield receipt.
    yield_resume_base: NearGas = None
    # Per byte cost of resume payload.
    yield_resume_byte: NearGas = None
