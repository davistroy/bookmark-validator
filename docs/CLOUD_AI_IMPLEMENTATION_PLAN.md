# Cloud AI Implementation Plan

## Overview

This document outlines the comprehensive plan for integrating Claude API and OpenAI API into the bookmark processor tool as alternatives to local AI processing. The implementation maintains backward compatibility while adding powerful cloud-based AI capabilities.

## Key Changes Summary

### 1. Platform Specification
- **Linux/WSL Only**: The tool is now explicitly Linux-only (including WSL2 on Windows)
- **No Windows Native Support**: Windows users must use WSL2

### 2. AI Engine Options
- **Local AI** (Default): Free, using Hugging Face transformers
- **Claude API**: Using Claude 3 Haiku model for cost-effectiveness
- **OpenAI API**: Using GPT-3.5-turbo model for cost-effectiveness

### 3. New Features
- Command-line option `--ai-engine` to select AI provider
- Configuration file support for API keys (never in code or CLI)
- Real-time cost tracking with user confirmation at $10 intervals
- Intelligent rate limiting for each API service
- Batch processing optimized for each provider
- Comprehensive error handling with fallback options

## Implementation Tasks (15 Total)

### Phase 1: Foundation (Tasks 1-3)
1. **Implement User Configuration System** (High Priority)
   - Create user_config.ini system
   - Auto-add to .gitignore
   - Secure API key storage
   - Configuration validation

2. **Update Command Line Interface** (High Priority)
   - Add `--ai-engine` parameter
   - Update `--help` with comprehensive docs
   - Validate engine selection
   - Display engine in verbose mode

3. **Create Base API Client Interface** (High Priority)
   - Abstract base class for all API clients
   - HTTP request handling with httpx
   - Connection pooling
   - Error handling and retry logic
   - Async context managers

### Phase 2: Core Implementation (Tasks 4-8)
4. **Implement Rate Limiter** (High Priority)
   - Service-specific rate limiters
   - Exponential backoff
   - Status reporting
   - Concurrent request tracking

5. **Implement Claude API Client** (Medium Priority)
   - Extend base API client
   - Claude 3 Haiku integration
   - Optimized prompts
   - Token tracking and cost calculation

6. **Implement OpenAI API Client** (Medium Priority)
   - Extend base API client
   - GPT-3.5-turbo integration
   - Optimized prompts
   - Token tracking and cost calculation

7. **Implement AI Factory and Selection Logic** (Medium Priority)
   - Factory pattern for AI client instantiation
   - Selection based on CLI arguments
   - API key validation
   - Fallback logic

8. **Implement Batch Processing** (Medium Priority)
   - Service-specific batch sizes
   - Concurrent processing within rate limits
   - Progress tracking per batch
   - Error handling for partial failures

### Phase 3: User Experience (Tasks 9-11)
9. **Implement Cost Tracking and User Control** (Medium Priority)
   - Running cost totals
   - User confirmation at $10 intervals
   - Cost limits from configuration
   - Detailed cost breakdown

10. **Implement Progress and Status Updates** (Low Priority)
    - Enhanced progress display
    - AI engine status
    - Rate limit status
    - Cost information
    - Final statistics

11. **Implement Error Handling and Fallbacks** (Medium Priority)
    - Fallback cascade: Cloud → Local → Content-based
    - User choice on failures
    - Error sanitization (no API keys in logs)
    - Continue processing after failures

### Phase 4: Optimization (Task 12)
12. **Optimize Prompts for Each AI Service** (Low Priority)
    - Service-specific prompt engineering
    - Token usage optimization
    - Response quality testing
    - Prompt selection system

### Phase 5: Security and Testing (Tasks 13-14)
13. **Implement Secure API Key Management** (High Priority)
    - Secure configuration loading
    - API key validation
    - Error message sanitization
    - No keys in logs or exceptions

14. **Implement Integration Tests** (Low Priority)
    - Real API testing with test keys
    - Mock responses for unit tests
    - Rate limiting behavior tests
    - Cost calculation accuracy tests
    - End-to-end processing tests

### Phase 6: Documentation (Task 15)
15. **Update Documentation and Create User Guide** (Low Priority)
    - Update README.md
    - Create API setup guide
    - Document cost tracking
    - Update help text
    - Add troubleshooting section

## Technical Specifications

### API Integration Details
- **HTTP Client**: httpx with async support
- **Rate Limits**: 
  - Claude: 50 requests/minute
  - OpenAI: 60 requests/minute
- **Batch Sizes**:
  - Claude: 10 bookmarks/batch
  - OpenAI: 20 bookmarks/batch
- **Timeouts**: 30 seconds default
- **Retries**: Exponential backoff, max 3 attempts

### Cost Information
- **Claude 3 Haiku**: ~$0.25 input / $1.25 output per million tokens
- **GPT-3.5-turbo**: ~$0.50 input / $1.50 output per million tokens
- **Estimated cost**: ~$0.001-0.002 per bookmark

### Security Requirements
- API keys only in configuration files
- Configuration files in .gitignore
- No API keys in:
  - Command-line arguments
  - Environment variables
  - Log files
  - Error messages
  - Git commits

## Configuration Example

```ini
[ai]
# Default engine: local, claude, or openai
default_engine = local

# Cloud AI configuration (NEVER commit this file!)
claude_api_key = sk-ant-api03-xxxx
openai_api_key = sk-xxxx

# Rate limiting (requests per minute)
claude_rpm = 50
openai_rpm = 60

# Batch sizes for cloud AI
claude_batch_size = 10
openai_batch_size = 20

# Cost tracking
show_running_costs = true
cost_confirmation_interval = 10.0  # USD
```

## Usage Examples

```bash
# Using local AI (default)
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv

# Using Claude API
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine claude

# Using OpenAI API
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine openai

# With verbose output showing costs
python -m bookmark_processor --input bookmarks.csv --output enhanced.csv --ai-engine claude --verbose
```

## Success Criteria

1. Both APIs fully integrated and functional
2. Cost tracking accurate to within 1%
3. Rate limiting prevents all API errors
4. Seamless switching between engines
5. API keys never exposed
6. All tests passing
7. Documentation complete
8. User can process 3,500+ bookmarks with cloud AI

## Timeline Estimate

- Phase 1 (Foundation): 2-3 days
- Phase 2 (Core Implementation): 4-5 days
- Phase 3 (User Experience): 2-3 days
- Phase 4 (Optimization): 1-2 days
- Phase 5 (Security & Testing): 2-3 days
- Phase 6 (Documentation): 1 day

**Total: 12-17 days**

## Notes for Implementation

1. Start with Phase 1 to establish the foundation
2. Test each API client thoroughly before moving to batch processing
3. Implement cost tracking early to avoid surprises
4. Focus on security throughout - API keys must never be exposed
5. Consider using environment-specific test API keys
6. Document any API-specific quirks or limitations discovered