pipeline {
    agent any

    environment {
        APP_NAME        = "journeyiq-fare-api"
        DOCKER_IMAGE    = "journeyiq/${APP_NAME}"
        DOCKER_TAG      = "${GIT_COMMIT[0..6]}-${BUILD_NUMBER}"
        STAGING_NS      = "journeyiq-staging"
        PROD_NS         = "journeyiq"
        PYTHON_VERSION  = "3.11"
    }

    options {
        timeout(time: 45, unit: "MINUTES")
        buildDiscarder(logRotator(numToKeepStr: "15"))
        disableConcurrentBuilds()                          // prevent race conditions
    }

    stages {

        //Checkout
        stage("Checkout") {
            steps {
                checkout scm
                script {
                    env.GIT_AUTHOR = sh(
                        script: "git log -1 --pretty=%an",
                        returnStdout: true
                    ).trim()
                    echo "Build triggered by: ${env.GIT_AUTHOR} | Branch: ${BRANCH_NAME}"
                }
            }
        }

        //Static Analysis
        stage("Static Analysis") {
            parallel {
                stage("Lint") {
                    steps {
                        sh """
                            pip install flake8 --quiet
                            flake8 . \
                                --max-line-length=110 \
                                --exclude=.git,__pycache__,migrations \
                                --count --statistics
                        """
                    }
                }
                stage("Security Scan (Bandit)") {
                    steps {
                        sh """
                            pip install bandit --quiet
                            bandit -r . -ll -x ./tests -f json -o bandit_report.json || true
                            cat bandit_report.json
                        """
                        archiveArtifacts artifacts: "bandit_report.json", allowEmptyArchive: true
                    }
                }
            }
        }

        //Unit Tests
        stage("Unit Tests") {
            steps {
                sh """
                    pip install -r requirements.txt --quiet
                    pip install pytest pytest-cov --quiet
                    pytest tests/ \
                        --cov=. \
                        --cov-report=xml:coverage.xml \
                        --cov-report=term-missing \
                        -v --tb=short
                """
                // Publish JUnit-format results (pytest generates these with --junit-xml)
                junit "test-results/*.xml"
                publishCoverage adapters: [coberturaAdapter("coverage.xml")]
            }
        }

        //Build Docker Image
        stage("Build Docker Image") {
            steps {
                script {
                    dockerImage = docker.build(
                        "${DOCKER_IMAGE}:${DOCKER_TAG}",
                        "--build-arg BUILD_DATE=${new Date().format('yyyy-MM-dd')} \
                         --build-arg GIT_SHA=${GIT_COMMIT[0..6]} \
                         --label git.branch=${BRANCH_NAME} \
                         ."
                    )
                    echo "Built image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
                }
            }
        }

        //Container Vulnerability Scan
        stage("Scan Image (Trivy)") {
            steps {
                sh """
                    docker run --rm \
                        -v /var/run/docker.sock:/var/run/docker.sock \
                        aquasec/trivy image \
                            --exit-code 1 \
                            --severity HIGH,CRITICAL \
                            --no-progress \
                            ${DOCKER_IMAGE}:${DOCKER_TAG}
                """
            }
        }

        //Push to Registry
        stage("Push to Registry") {
            when { branch "main" }
            steps {
                script {
                    docker.withRegistry("https://registry.hub.docker.com",
                                        env.DOCKER_CREDENTIALS_ID) {
                        dockerImage.push(DOCKER_TAG)
                        dockerImage.push("latest")
                    }
                    echo "Pushed ${DOCKER_IMAGE}:${DOCKER_TAG} to Docker Hub"
                }
            }
        }

        //Deploy to Staging
        stage("Deploy to Staging") {
            when { branch "main" }
            steps {
                withCredentials([file(credentialsId: env.KUBECONFIG_CREDENTIAL,
                                      variable: "KUBECONFIG")]) {
                    sh """
                        kubectl set image deployment/journeyiq-fare-api \
                            fare-api=${DOCKER_IMAGE}:${DOCKER_TAG} \
                            -n ${STAGING_NS}
                        kubectl rollout status deployment/journeyiq-fare-api \
                            -n ${STAGING_NS} --timeout=120s
                    """
                }
            }
        }

        //Smoke Test
        stage("Smoke Test (Staging)") {
            when { branch "main" }
            steps {
                sh """
                    STAGING_HOST=\$(kubectl get svc journeyiq-fare-api-external \
                        -n ${STAGING_NS} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
                    HTTP_STATUS=\$(curl -s -o /dev/null -w '%{http_code}' \
                        http://\${STAGING_HOST}/api/v1/health)
                    if [ "\$HTTP_STATUS" != "200" ]; then
                        echo "Smoke test FAILED — status: \${HTTP_STATUS}"
                        exit 1
                    fi
                    echo "Smoke test PASSED — status: \${HTTP_STATUS}"
                """
            }
        }

        //Deploy to Production
        stage("Deploy to Production") {
            when { branch "main" }
            input {
                message "Deploy ${DOCKER_IMAGE}:${DOCKER_TAG} to PRODUCTION?"
                ok      "Deploy"
                submitter "mlops-lead,tech-lead"
                parameters {
                    booleanParam(name: "CONFIRM", defaultValue: false,
                                 description: "Check to confirm production deployment")
                }
            }
            steps {
                script {
                    if (!params.CONFIRM) {
                        error("Production deployment not confirmed. Aborting.")
                    }
                }
                withCredentials([file(credentialsId: env.KUBECONFIG_CREDENTIAL,
                                      variable: "KUBECONFIG")]) {
                    sh """
                        kubectl set image deployment/journeyiq-fare-api \
                            fare-api=${DOCKER_IMAGE}:${DOCKER_TAG} \
                            -n ${PROD_NS}
                        kubectl rollout status deployment/journeyiq-fare-api \
                            -n ${PROD_NS} --timeout=180s
                    """
                }
            }
        }
    }

    //Post-actions
    post {
        always {
            cleanWs()
        }
        success {
            echo "Pipeline SUCCEEDED for ${DOCKER_IMAGE}:${DOCKER_TAG}"
            sh """
                curl -s -X POST ${SLACK_WEBHOOK_URL} \
                    -H 'Content-type: application/json' \
                    -d '{"text":"✅ JourneyIQ build #${BUILD_NUMBER} PASSED | Branch: ${BRANCH_NAME} | Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"}'
            """
        }
        failure {
            echo "Pipeline FAILED at stage: ${currentBuild.result}"
            sh """
                curl -s -X POST ${SLACK_WEBHOOK_URL} \
                    -H 'Content-type: application/json' \
                    -d '{"text":"❌ JourneyIQ build #${BUILD_NUMBER} FAILED | Branch: ${BRANCH_NAME} | Check: ${BUILD_URL}"}'
            """ 
        }
    }
}
