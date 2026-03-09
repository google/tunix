/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @param {{github: !Object, context: !Object}} params
 * @returns {!Promise<void>}
 */
module.exports = async ({github, context}) => {
  let issueNumber;
  let assigneesList;

  if (context.payload.issue) {
    assigneesList = [
      "tianshub",
      "wang2yn84",
      "lc5211",
      "hgao327",
      "sizhit2",
      "abheesht17",
      "jiangyangmu"
    ];  // for issues
    issueNumber = context.payload.issue.number;
  } else if (context.payload.pull_request) {
    assigneesList = [
      "tianshub",
      "wang2yn84",
      "lc5211",
      "hgao327",
      "sizhit2",
      "abheesht17",
      "jiangyangmu"
    ];  // for PRs
    issueNumber = context.payload.pull_request.number;
  } else {
    console.log('Not an issue or PR');
    return;
  }

  console.log('Assignee list:', assigneesList);
  console.log('Entered auto assignment for this issue/PR:', issueNumber);

  if (!assigneesList.length) {
    console.log('No assignees found for this repo.');
    return;
  }

  // To assign issue or PRs on Weekly Rotation basis
  const now = new Date();
  // Calculate total weeks since Unix Epoch (Jan 1, 1970)
  // 604800000 is the number of milliseconds in a week
  const weekCount = Math.floor(now.getTime() / 604800000);

  const noOfAssignees = assigneesList.length;
  const selection = weekCount % noOfAssignees;
  const assigneeForIssue = assigneesList[selection];

  console.log(
      `Issue/PR Number = ${issueNumber}, assigning to: ${assigneeForIssue}`);

  return github.rest.issues.addAssignees({
    issue_number: issueNumber,
    owner: context.repo.owner,
    repo: context.repo.repo,
    assignees: [assigneeForIssue],
  });
};
