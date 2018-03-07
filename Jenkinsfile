pipeline {
    agent { label 'docker' }

    environment {
        COLLECTOR_IMAGE = "gros-data-analysis-dashboard"
        PREDICTOR_TAG = env.BRANCH_NAME.replaceFirst('^master$', 'latest')
        PREDICTOR_NAME = "gros-prediction"
        PREDICTOR_IMAGE = "${env.PREDICTOR_NAME}:${env.PREDICTOR_TAG}"
        GITLAB_TOKEN = credentials('prediction-gitlab-token')
    }

    parameters {
        string(name: 'PREDICTION_ARGS', defaultValue: '--label "round(num_story_points)" --roll-sprints 3 --roll-labels --replace-na --model abe --test-interval 100 --num-epochs 100 --no-stratified-split --learning-rate 0.1 --test-size 0.2 --assign "over_value=velocity>avg_velocity" --assign "over_expectation=num_story_points-initial_story_points" --index sprint_days,number_of_vcs_devs,initial_story_points,num_weighted_points,num_links,num_comments,avg_concurrent_progress --keep-index initial_story_points --exponent 2 --distance cosine', description: 'Prediction arguments')
        string(name: 'PREDICTION_ORGANIZATIONS', defaultValue: "${env.PREDICTION_ORGANIZATIONS}", description: 'Organizations to include in prediction')
    }
    options {
        gitLabConnection('gitlab')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    triggers {
        gitlab(triggerOnPush: true, triggerOnMergeRequest: true, branchFilterType: 'All', secretToken: env.GITLAB_TOKEN)
        cron(env.PREDICTION_CRON)
    }

    post {
        success {
            archiveArtifacts artifacts: 'output/**/*.json,output/sprint_features.arff,schema/**/*.json', onlyIfSuccessful: true
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'success'
        }
        failure {
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'failed'
        }
        aborted {
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'canceled'
        }
    }

    stages {
        stage('Build') {
            steps {
                checkout scm
                sh 'docker build -t $DOCKER_REPOSITORY/$PREDICTOR_IMAGE . --build-arg TENSORFLOW_VERSION=$TENSORFLOW_VERSION'
            }
        }
        stage('Push') {
            steps {
                withDockerRegistry(credentialsId: 'docker-credentials', url: env.DOCKER_URL) {
                    sh 'docker push $DOCKER_REPOSITORY/$PREDICTOR_IMAGE'
                }
            }
        }
        stage('Early exit') {
            when {
                not {
                    triggeredBy 'TimerTrigger'
                }
                not {
                    triggeredBy 'UserIdCause'
                }
                branch pattern: ".*time-machine\$", comparator: "REGEXP"
            }
            steps {
                script {
                    currentBuild.getRawBuild().getExecutor().interrupt(Result.UNSTABLE)
                    sleep(1)
                }
            }
        }
        stage('Collect') {
            agent {
                docker {
                    image "${env.COLLECTOR_IMAGE}"
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh "/bin/bash -cex \"rm -rf \$WORKSPACE/output; mkdir \$WORKSPACE/output; cd /home/docker; for org in ${params.PREDICTION_ORGANIZATIONS}; do Rscript features.r --core --log INFO --config $ANALYSIS_CONFIGURATION $REPORT_PARAMS --output \$WORKSPACE/output --filename sprint_features.${env.BUILD_TAG}.arff --append --org \\\$org; done\""
                }
            }
        }
        stage('Upload') {
            when {
                beforeAgent true
                environment name: 'PREDICTOR_REMOTE', value: 'true'
            }
            agent {
                docker {
                    image "${env.PREDICTOR_IMAGE}"
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    args '-v /usr/local/share/ca-certificates/:/usr/local/share/ca-certificates/'
                    label 'master'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "python files.py --upload output/sprint_features.${env.BUILD_TAG}.arff --config $PREDICTOR_CONFIGURATION"
                }
            }
        }
        stage('Predict') {
            when {
                beforeAgent true
                not { environment name: 'PREDICTOR_REMOTE', value: 'true' }
            }
            agent {
                docker {
                    image "${env.PREDICTOR_IMAGE}"
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    reuseNode true
                }
            }
            steps {
                sh "python tensor.py --filename output/sprint_features.${env.BUILD_TAG}.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.${env.BUILD_TAG}.json"
            }
        }
        stage('Predict remote GPU') {
            when {
                beforeAgent true
                environment name: 'PREDICTOR_REMOTE', value: 'true'
            }
            agent {
                docker {
                    image "${env.PREDICTOR_IMAGE}"
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    args '--runtime=nvidia -v /usr/local/share/ca-certificates/:/usr/local/share/ca-certificates/'
                    label 'gpu'
                    alwaysPull true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "CUDA_VISIBLE_DEVICES=${env.EXECUTOR_NUMBER} python tensor.py --filename output/sprint_features.${env.BUILD_TAG}.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.${env.BUILD_TAG}.json --store owncloud --device /gpu:${env.EXECUTOR_NUMBER} --config $PREDICTOR_CONFIGURATION"
                }
            }
        }
        stage('Download') {
            when {
                beforeAgent true
                environment name: 'PREDICTOR_REMOTE', value: 'true'
            }
            agent {
                docker {
                    image "${env.PREDICTOR_IMAGE}"
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    args '-v /usr/local/share/ca-certificates/:/usr/local/share/ca-certificates/'
                    label 'master'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "python files.py --download output/sprint_labels.${env.BUILD_TAG}.json --remove output/sprint_features.${env.BUILD_TAG}.arff output/sprint_labels.${env.BUILD_TAG}.json --config $PREDICTOR_CONFIGURATION"
                }
            }
        }
        stage('Format') {
            agent {
                docker {
                    image "${env.COLLECTOR_IMAGE}"
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh "/bin/bash -cex \"cd /home/docker; for org in ${params.PREDICTION_ORGANIZATIONS}; do Rscript sprint_results.r --file \$WORKSPACE/output/sprint_labels.${env.BUILD_TAG}.json --features \$WORKSPACE/output/sprint_features.${env.BUILD_TAG}.arff --config $ANALYSIS_CONFIGURATION --output \$WORKSPACE/output $REPORT_PARAMS --org \\\$org; if [ -e \$WORKSPACE/output/\\\$org/descriptions.json ]; then cp \$WORKSPACE/output/\\\$org/descriptions.json \$WORKSPACE/output/; fi; done\""
                    sh "mv output/sprint_labels.${env.BUILD_TAG}.json output/sprint_labels.json"
                    sh "mv output/sprint_features.${env.BUILD_TAG}.arff output/sprint_features.arff"
                }
            }
        }
    }
}
