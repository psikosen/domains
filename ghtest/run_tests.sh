#!/bin/bash

# run_tests.sh
# Master test runner for Latent Space OS validation

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║          LATENT SPACE OS - TEST SUITE RUNNER            ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Install with:"
    echo "   pip install torch numpy matplotlib"
    exit 1
fi
echo "✓ Dependencies OK"
echo ""

# Run mathematical validation tests
echo "════════════════════════════════════════════════════════════"
echo "PHASE 1: Mathematical Validation"
echo "════════════════════════════════════════════════════════════"
python3 test_math_validation.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Mathematical validation failed!"
    exit 1
fi

echo ""
echo ""

# Run component integration tests
echo "════════════════════════════════════════════════════════════"
echo "PHASE 2: Component Integration"
echo "════════════════════════════════════════════════════════════"
python3 test_component_integration.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Component integration failed!"
    exit 1
fi

echo ""
echo ""

# Final summary
echo "════════════════════════════════════════════════════════════"
echo "TEST SUITE SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "✅ Phase 1: Mathematical Validation - PASSED"
echo "   • Holographic capacity bounds verified"
echo "   • Sparse autoencoder reconstruction validated"
echo "   • Content-addressable memory precision confirmed"
echo "   • Vector arithmetic for reasoning works"
echo "   • Gradient stability proven"
echo "   • Hierarchical indexing speedup observed"
echo ""
echo "✅ Phase 2: Component Integration - PASSED"
echo "   • Read-write cycle consistency verified"
echo "   • Multi-concept storage working"
echo "   • Atomic updates functional"
echo "   • Thought chains converge"
echo "   • Memory persistence validated"
echo "   • Kernel protection active"
echo "   • Sparse feature activation correct"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "🎉 ALL TESTS PASSED - SYSTEM VALIDATED 🎉"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Run the full prototype: python3 prototype.py"
echo "  2. Review results in each test file"
echo "  3. Start implementing Phase 1 from roadmap"
echo ""
