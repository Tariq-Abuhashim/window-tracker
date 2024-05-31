# widnow_tracker

<p align='center'>
    <img src="media/lines.png" alt="drawing" width="500"/>
    <img src="media/normals.png" alt="drawing" width="440"/>
</p>

## Description

This is the window global normal computation code. The code present in this repository performs the following tasks:  
1- Build a 3D line sketch using all image frames, poses and 3D points from either COLMAP or ORBSLAM3.  
2- Detect windows using DETR.   
3- Track windows in image frames using a combination of 2D line tracks, 2D bounding box detections and heuristics.    
4- Using all accommulated line tracks per window object, it computes the normal direction and window centroid.  

Main dependencies include: 
1- COLMAP (https://github.com/colmap/colmap).  
2- PoseLib (https://github.com/PoseLib/PoseLib).  
3- ORBSLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3).  
4- DETR (https://gitlab.com/missionsystems/hyperteaming/detr).  
Depending on if either COLMAP or ORBSLAM3 are used, some of these dependencies can be ignored.  


## Getting started

Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

```
cd existing_repo
git remote add origin https://gitlab.com/missionsystems/hyperteaming/widnow_tracker.git
git branch -M master
git push -uf origin master
```
## Usage

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Authors

Tariq Abuhashim for mission-systems, May, 2024

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/missionsystems/hyperteaming/widnow_tracker/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***
