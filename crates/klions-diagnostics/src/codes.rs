//! Stable diagnostic code registry.
//!
//! FR-ERR-003: a code, once assigned, is never reused for a different error
//! class. FR-ERR-005: AI-specific classes occupy distinct ranges.
//!
//! ```text
//!   K0xxx  general: lexical, syntactic, type, name, runtime
//!   K1xxx  ShapeError
//!   K2xxx  DatasetError
//!   K3xxx  TrainingError
//!   K4xxx  ModelError
//!   K5xxx  DeviceError
//!   K9xxx  internal compiler error
//! ```

// ---------- K0xxx — lexical (001-019) ----------
pub const INVALID_UTF8: &str = "K0001";
pub const UNTERMINATED_STRING: &str = "K0002";
pub const UNTERMINATED_BLOCK_COMMENT: &str = "K0003";
pub const UNKNOWN_ESCAPE: &str = "K0004";
pub const UNEXPECTED_CHARACTER: &str = "K0005";
pub const NON_ASCII_IDENTIFIER: &str = "K0006";
pub const RESERVED_KEYWORD: &str = "K0007";
pub const MALFORMED_NUMBER: &str = "K0008";
pub const INVALID_UNICODE_ESCAPE: &str = "K0009";

// ---------- K0xxx — syntax (020-039) ----------
pub const UNEXPECTED_TOKEN: &str = "K0020";
pub const EXPECTED_TOKEN: &str = "K0021";
pub const UNCLOSED_DELIMITER: &str = "K0022";
pub const UNKNOWN_LAYER: &str = "K0023";
pub const UNKNOWN_TRAIN_OPTION: &str = "K0024";
pub const POSITIONAL_AFTER_NAMED: &str = "K0025";
pub const BAD_IMPORT_PATH: &str = "K0026";
pub const BAD_TENSOR_TYPE: &str = "K0027";
pub const EXPECTED_EXPRESSION: &str = "K0028";
pub const EXPECTED_STATEMENT: &str = "K0029";

// ---------- K0xxx — names & types (040-079) ----------
pub const UNDEFINED_NAME: &str = "K0040";
pub const DUPLICATE_DEFINITION: &str = "K0041";
pub const TYPE_MISMATCH: &str = "K0042";
pub const NO_IMPLICIT_CONVERSION: &str = "K0043";
pub const BAD_CAST: &str = "K0044";
pub const NOT_CALLABLE: &str = "K0045";
pub const WRONG_ARG_COUNT: &str = "K0046";
pub const UNKNOWN_NAMED_ARG: &str = "K0047";
pub const MISSING_RETURN: &str = "K0048";
pub const RETURN_TYPE_MISMATCH: &str = "K0049";
pub const ASSIGN_TO_IMMUTABLE: &str = "K0050";
pub const NOT_INDEXABLE: &str = "K0051";
pub const NO_SUCH_FIELD: &str = "K0052";
pub const QUESTION_OUTSIDE_RESULT: &str = "K0053";
pub const QUESTION_ERROR_MISMATCH: &str = "K0054";
pub const CONDITION_NOT_BOOL: &str = "K0055";
pub const BREAK_OUTSIDE_LOOP: &str = "K0056";
pub const CONTINUE_OUTSIDE_LOOP: &str = "K0057";
pub const NOT_ITERABLE: &str = "K0058";
pub const UNUSED_VARIABLE: &str = "K0059";
pub const UNUSED_PARAMETER: &str = "K0060";
pub const FORMAT_ARG_COUNT: &str = "K0061";
pub const BAD_OPERAND_TYPE: &str = "K0062";
pub const MISSING_MAIN: &str = "K0063";
pub const NOT_A_MODEL: &str = "K0064";
pub const NOT_A_DATASET: &str = "K0065";
pub const MISSING_TRAIN_OPTION: &str = "K0066";
pub const ANNOTATION_MISMATCH: &str = "K0067";

// ---------- K0xxx — runtime (080-099) ----------
pub const STACK_OVERFLOW: &str = "K0080";
pub const DIVISION_BY_ZERO: &str = "K0081";
pub const INTEGER_OVERFLOW: &str = "K0082";
pub const INDEX_ERROR: &str = "K0083";
pub const ASSERTION_FAILED: &str = "K0084";
pub const UNWRAP_ON_ERR: &str = "K0085";
pub const RUNTIME_PANIC: &str = "K0086";
pub const IO_ERROR: &str = "K0087";

// ---------- K1xxx — ShapeError ----------
pub const SHAPE_MISMATCH: &str = "K1001";
pub const MATMUL_SHAPE: &str = "K1002";
pub const RESHAPE_SIZE: &str = "K1003";
pub const BAD_RANK: &str = "K1004";
pub const BAD_AXIS: &str = "K1005";
pub const BROADCAST_FAILED: &str = "K1006";
pub const LAYER_CHAIN_MISMATCH: &str = "K1007";
pub const MODEL_INPUT_MISMATCH: &str = "K1008";
pub const NEGATIVE_DIMENSION: &str = "K1009";

// ---------- K2xxx — DatasetError ----------
pub const DATASET_NOT_FOUND: &str = "K2001";
pub const IDX_BAD_MAGIC: &str = "K2002";
pub const IDX_TRUNCATED: &str = "K2003";
pub const SAMPLE_COUNT_MISMATCH: &str = "K2004";
pub const CSV_MALFORMED: &str = "K2005";
pub const CSV_BAD_LABEL_COLUMN: &str = "K2006";
pub const DATASET_EMPTY: &str = "K2007";
pub const PATH_TRAVERSAL: &str = "K2008";

// ---------- K3xxx — TrainingError ----------
pub const NON_FINITE_GRADIENT: &str = "K3001";
pub const BAD_HYPERPARAMETER: &str = "K3002";
pub const NO_PARAMETERS: &str = "K3003";
pub const UNKNOWN_LOSS: &str = "K3004";
pub const UNKNOWN_METRIC: &str = "K3005";
pub const NO_GRAD_PATH: &str = "K3006";
pub const BATCH_TOO_LARGE: &str = "K3007";

// ---------- K4xxx — ModelError ----------
pub const MODEL_RECURSIVE: &str = "K4001";
pub const MODEL_EMPTY: &str = "K4002";
pub const MODEL_FILE_BAD_MAGIC: &str = "K4003";
pub const MODEL_FILE_VERSION: &str = "K4004";
pub const MODEL_SHAPE_MANIFEST: &str = "K4005";

// ---------- K5xxx — DeviceError ----------
pub const UNSUPPORTED_DEVICE: &str = "K5001";
pub const ALLOCATION_CEILING: &str = "K5002";
pub const OUT_OF_MEMORY: &str = "K5003";

// ---------- K6xxx — deferred features (FR-ERR-010) ----------
pub const DEFERRED_FEATURE: &str = "K6001";

// ---------- K9xxx — internal ----------
pub const INTERNAL_ERROR: &str = "K9001";

/// Human-readable class for a code, used by `--json` output and docs.
pub fn class_of(code: &str) -> &'static str {
    match code.as_bytes().get(1) {
        Some(b'1') => "ShapeError",
        Some(b'2') => "DatasetError",
        Some(b'3') => "TrainingError",
        Some(b'4') => "ModelError",
        Some(b'5') => "DeviceError",
        Some(b'6') => "DeferredFeature",
        Some(b'9') => "InternalError",
        _ => "CompileError",
    }
}
