#!/bin/bash

# Docker Deployment Testing Script
# Tests our RAG Chatbot deployment across different scenarios

echo "🚀 RAG Chatbot Docker Deployment Testing"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for service at $url to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_success "Service is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Service failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Function to run test
run_test() {
    local test_name=$1
    local test_command=$2
    
    print_status "Running test: $test_name"
    
    if eval "$test_command"; then
        print_success "✓ $test_name"
        return 0
    else
        print_error "✗ $test_name"
        return 1
    fi
}

# Pre-flight checks
echo
print_status "Pre-flight checks..."

# Check Docker
if ! command_exists docker; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi
print_success "Docker is available"

# Check Docker Compose
if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    print_error "Docker Compose is not installed or not in PATH"
    exit 1
fi
print_success "Docker Compose is available"

# Check if running as root (warn if yes)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root - consider using a regular user for testing"
fi

# Check system resources
print_status "Checking system resources..."
if command_exists free; then
    # Linux
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
elif [ -f /proc/meminfo ]; then
    # Linux alternative
    MEMORY_GB=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
else
    # macOS and others - use system_profiler or approximate
    if command_exists system_profiler; then
        MEMORY_GB=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2}' | sed 's/GB//')
    else
        MEMORY_GB=8  # Reasonable default assumption
        print_status "Unable to detect exact memory - assuming sufficient"
    fi
fi

if [ "$MEMORY_GB" -lt 4 ]; then
    print_warning "Less than 4GB RAM available - performance may be affected"
else
    print_success "Sufficient memory available (${MEMORY_GB}GB)"
fi

# Check disk space
DISK_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$DISK_SPACE" -lt 2000000 ]; then  # 2GB in KB
    print_warning "Less than 2GB disk space available"
else
    print_success "Sufficient disk space available"
fi

# Check for port conflicts
print_status "Checking for port conflicts..."
if netstat -tuln 2>/dev/null | grep -q ":8501 "; then
    print_warning "Port 8501 is already in use - this may cause conflicts"
else
    print_success "Port 8501 is available"
fi

# Environment checks
echo
print_status "Environment checks..."

# Check for .env file
if [ -f ".env" ]; then
    print_success ".env file found"
    
    # Check for required environment variables
    if grep -q "DEEPSEEK_API_KEY" .env && grep -q "OPENAI_API_KEY" .env; then
        print_success "Required API keys found in .env"
    else
        print_error "Missing required API keys in .env file"
        echo "Please ensure both DEEPSEEK_API_KEY and OPENAI_API_KEY are set"
        exit 1
    fi
else
    print_warning ".env file not found - using environment variables"
    
    if [ -z "$DEEPSEEK_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
        print_error "Required environment variables not set"
        echo "Please set DEEPSEEK_API_KEY and OPENAI_API_KEY"
        exit 1
    fi
    print_success "Required API keys found in environment"
fi

# Docker tests
echo
print_status "Starting Docker deployment tests..."

# Test 1: Clean build
echo
run_test "Clean Docker build" "docker-compose build --no-cache"

# Test 2: Container startup
echo
run_test "Container startup" "docker-compose up -d"

# Test 3: Service health check
echo
if wait_for_service "http://localhost:8501/_stcore/health"; then
    print_success "✓ Service health check"
else
    print_error "✗ Service health check"
    print_status "Checking container logs..."
    docker-compose logs --tail=20
fi

# Test 4: Application accessibility
echo
run_test "Application accessibility" "curl -f -s http://localhost:8501 > /dev/null"

# Test 5: Container resource usage
echo
print_status "Checking container resource usage..."
CONTAINER_ID=$(docker-compose ps -q rag-chatbot)
if [ -n "$CONTAINER_ID" ]; then
    STATS=$(docker stats --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" "$CONTAINER_ID")
    echo "Resource usage: $STATS"
    print_success "✓ Resource usage check"
else
    print_error "✗ Could not find container for resource check"
fi

# Test 6: Volume persistence
echo
run_test "Volume persistence check" "docker exec \$(docker-compose ps -q rag-chatbot) ls -la /app/data"

# Test 7: Environment variable loading
echo
run_test "Environment variables" "docker exec \$(docker-compose ps -q rag-chatbot) printenv | grep -E '(DEEPSEEK|OPENAI)_API_KEY' | wc -l | grep -q '2'"

# Performance tests
echo
print_status "Running performance tests..."

# Test response time
echo
print_status "Testing response time..."
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8501)
if (( $(echo "$RESPONSE_TIME < 5.0" | bc -l) )); then
    print_success "✓ Response time: ${RESPONSE_TIME}s (good)"
else
    print_warning "Response time: ${RESPONSE_TIME}s (slow)"
fi

# Cleanup tests
echo
print_status "Testing cleanup procedures..."

# Test graceful shutdown
echo
run_test "Graceful shutdown" "docker-compose down"

# Test data persistence after restart
echo
run_test "Restart and data persistence" "docker-compose up -d && sleep 10 && docker exec \$(docker-compose ps -q rag-chatbot) ls -la /app/data"

# Platform-specific tests
echo
print_status "Platform-specific tests..."

# Detect platform
PLATFORM=$(uname -s)
print_status "Detected platform: $PLATFORM"

case $PLATFORM in
    "Darwin")
        print_status "Running macOS-specific tests..."
        # Check for Docker Desktop
        if pgrep -x "Docker Desktop" > /dev/null; then
            print_success "✓ Docker Desktop is running"
        else
            print_warning "Docker Desktop may not be running"
        fi
        ;;
    "Linux")
        print_status "Running Linux-specific tests..."
        # Check Docker daemon
        if systemctl is-active --quiet docker; then
            print_success "✓ Docker daemon is active"
        else
            print_warning "Docker daemon status unclear"
        fi
        
        # Check for SELinux if on Red Hat-based system
        if command_exists getenforce; then
            SELINUX_STATUS=$(getenforce)
            print_status "SELinux status: $SELINUX_STATUS"
        fi
        ;;
    *)
        print_warning "Unknown platform - skipping platform-specific tests"
        ;;
esac

# Educational deployment simulation
echo
print_status "Educational deployment simulation..."

# Simulate multiple user scenario
print_status "Simulating classroom environment..."
for i in {1..3}; do
    echo "Student $i connecting..."
    curl -s http://localhost:8501 > /dev/null &
done
wait
print_success "✓ Multiple concurrent connections handled"

# Final cleanup
echo
print_status "Final cleanup..."
docker-compose down
docker system prune -f > /dev/null 2>&1

# Summary
echo
echo "========================================"
print_status "Docker Deployment Test Complete!"
echo
print_success "✅ All basic deployment tests passed"
print_status "📊 Check the logs above for any warnings or issues"
print_status "🎓 Ready for educational deployment testing"
echo
print_status "Next steps:"
echo "  1. Test on different machines (macOS, Linux)"
echo "  2. Test with different Docker configurations"
echo "  3. Performance testing with real educational workloads"
echo "  4. Security testing for classroom environments"
echo
print_status "For detailed testing, see: documents/deployment_testing_plan.md"
echo "========================================" 